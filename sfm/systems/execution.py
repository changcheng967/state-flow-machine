"""
System 2: Execution (the breakthrough)

State Slot Bank: 64 registers that explicitly bind to variables and track
values through execution. Uses DeltaNet recurrent cell with eigenvalues
in [-1, 1] (not [0, 1]).

CUBE-OPTIMIZED:
- ALL statements processed in parallel where possible
- Embed all statements at once: (batch, num_statements, d_model) — one Cube call
- Compute all slot-match scores at once: (batch, num_statements, num_slots) — one Cube call
- DeltaNet uses matrix-based parallel scan over full sequence
- Only slot WRITE-BACK must be sequential (each write depends on previous state)
- Write-back batches across slot dimension (64 slots in parallel per statement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.components.state_slots import StateSlotBank
from sfm.components.deltanet_cell import DeltaNetStack


class ExecutionSystem(nn.Module):
    """
    System 2: Execution - CUBE-OPTIMIZED VERSION

    State Slot Bank with 64 registers that explicitly bind to variables
    and track values through execution.

    Key features:
    - Full sequence batched processing (parallel embedding, attention)
    - DeltaNet with eigenvalues in [-1, 1] for state tracking
    - Explicit variable binding via attention
    - All dimensions multiples of 16 for Cube unit
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,      # Multiple of 16
        num_slots: int = 16,         # Default 16 for exp0 (multiple of 16)
        slot_dim: int = 128,         # Multiple of 16
        max_ticks: int = 2,
        num_heads: int = 4,          # head_dim = 32
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Verify dimension alignment
        assert hidden_dim % 16 == 0, "hidden_dim must be multiple of 16"
        assert num_slots % 16 == 0, "num_slots must be multiple of 16"
        assert slot_dim % 16 == 0, "slot_dim must be multiple of 16"

        # State Slot Bank - main processing
        self.slot_bank = StateSlotBank(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_heads=num_heads,
            max_ticks=max_ticks,
            halting_threshold=0.5,
            dropout=dropout
        )

        # DeltaNet for additional sequential processing
        self.deltanet = DeltaNetStack(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout
        )

        # Variable binding attention (Cube-optimized)
        self.binding_proj = nn.Linear(input_dim, slot_dim)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        # Combine projections (input_dim + slot_dim -> input_dim)
        # Pad to next multiple of 16 if needed
        combined_dim = input_dim + slot_dim
        self.combine_proj = nn.Linear(combined_dim, input_dim)

        # Project deltanet output back to input_dim
        self.deltanet_proj = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()

        # Skip connection: direct path for simple patterns
        self.input_skip = nn.Linear(input_dim, input_dim, bias=False)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout)
        )

        self.scale = slot_dim ** -0.5

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """Initialize execution state."""
        slots = self.slot_bank.init_slots(batch_size, device, dtype)
        return {
            "slots": slots,
            "deltanet_state": None
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass - CUBE-OPTIMIZED with batched operations.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            state: Optional execution state.
            return_state: Whether to return updated state.

        Returns:
            Output tensor, optionally with state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device, dtype)

        slots = state["slots"]
        deltanet_state = state["deltanet_state"]

        # CUBE: Process entire sequence through slot bank at once
        # (batch, seq_len, input_dim) -> (batch, seq_len, input_dim)
        slot_out, slots, slot_info = self.slot_bank(x, slots)

        # CUBE: Process through DeltaNet with matrix-based parallel scan
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim) -> (batch, seq_len, input_dim)
        deltanet_out, deltanet_state = self.deltanet(x, deltanet_state)
        deltanet_out = self.deltanet_proj(deltanet_out)

        # CUBE: Compute variable binding attention for entire sequence at once
        # queries: (batch, seq_len, slot_dim)
        # keys: (batch, num_slots, slot_dim)
        queries = self.binding_proj(x)
        keys = self.slot_keys.unsqueeze(0).expand(batch_size, -1, -1)

        # CUBE: Batched matmul for attention scores
        # (batch, seq_len, slot_dim) @ (batch, slot_dim, num_slots) -> (batch, seq_len, num_slots)
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # CUBE: Weighted slot combination
        # (batch, seq_len, num_slots) @ (batch, num_slots, slot_dim) -> (batch, seq_len, slot_dim)
        bindings = torch.bmm(attn_weights, slots)

        # CUBE: Combine all outputs (concatenation + linear)
        # (batch, seq_len, input_dim + slot_dim)
        combined = torch.cat([slot_out, bindings], dim=-1)
        output = self.combine_proj(combined)

        # Add deltanet output (residual) and input skip connection
        output = output + deltanet_out + self.input_skip(x)

        # Final projection
        output = self.output_proj(output)

        new_state = {
            "slots": slots,
            "deltanet_state": deltanet_state
        }

        info = {
            "avg_ticks_per_token": slot_info.get("avg_ticks", 1.0)
        }

        if return_state:
            return output, new_state, info
        return output

    def get_slot_values(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get current slot values for inspection."""
        return state["slots"]

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Execution System (Cube-Optimized) Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)
    import time

    batch_size = 4
    seq_len = 16
    input_dim = 256   # Multiple of 16
    hidden_dim = 256  # Multiple of 16
    num_slots = 16    # Multiple of 16
    slot_dim = 128    # Multiple of 16
    max_ticks = 2

    # Verify dimensions
    print("\n1. Verifying dimension alignment...")
    assert input_dim % 16 == 0, f"input_dim {input_dim} not multiple of 16"
    assert hidden_dim % 16 == 0, f"hidden_dim {hidden_dim} not multiple of 16"
    assert num_slots % 16 == 0, f"num_slots {num_slots} not multiple of 16"
    assert slot_dim % 16 == 0, f"slot_dim {slot_dim} not multiple of 16"
    print(f"   input_dim: {input_dim} (x{input_dim // 16})")
    print(f"   hidden_dim: {hidden_dim} (x{hidden_dim // 16})")
    print(f"   num_slots: {num_slots} (x{num_slots // 16})")
    print(f"   slot_dim: {slot_dim} (x{slot_dim // 16})")
    print("   [OK] Dimension alignment verified!")

    # Initialize execution system
    print("\n2. Initializing ExecutionSystem...")
    execution = ExecutionSystem(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_slots=num_slots,
        slot_dim=slot_dim,
        max_ticks=max_ticks,
        num_heads=4
    )

    # Test forward pass
    print("\n3. Testing Cube-optimized forward pass...")
    x = torch.randn(batch_size, seq_len, input_dim)

    start = time.time()
    output, state, info = execution.forward(x, return_state=True)
    elapsed = time.time() - start

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Forward pass time: {elapsed*1000:.2f}ms")
    print(f"   Average ticks: {info['avg_ticks_per_token']:.2f}")
    assert output.shape == x.shape, "Output shape mismatch!"
    assert "slots" in state, "State missing slots!"
    print("   [OK] Forward pass test passed!")

    # Performance test
    print("\n4. Performance test (100 iterations)...")
    start = time.time()
    for _ in range(100):
        output, state, _ = execution.forward(x, state=state, return_state=True)
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({elapsed*10:.1f}ms per iter)")

    # Test state persistence
    print("\n5. Testing state persistence...")
    x2 = torch.randn(batch_size, seq_len, input_dim)
    output2, state2, info2 = execution.forward(x2, state=state, return_state=True)
    print(f"   Second pass output shape: {output2.shape}")
    assert output2.shape == x2.shape, "Second output shape mismatch!"
    print("   [OK] State persistence test passed!")

    # Test slot inspection
    print("\n6. Testing slot inspection...")
    slot_values = execution.get_slot_values(state2)
    print(f"   Slot values shape: {slot_values.shape}")
    assert slot_values.shape == (batch_size, num_slots, slot_dim), "Slot values shape mismatch!"
    print("   [OK] Slot inspection test passed!")

    # Test gradient flow
    print("\n7. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for param in execution.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   [OK] Gradient flow test passed!")

    # Count parameters
    total_params = execution.count_parameters()
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All Execution System (Cube-Optimized) tests passed!")
    print("=" * 60)
