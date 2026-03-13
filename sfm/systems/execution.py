"""
System 2: Execution (the breakthrough)

State Slot Bank: 64 registers that explicitly bind to variables and track
values through execution. Uses DeltaNet recurrent cell with eigenvalues
in [-1, 1] (not [0, 1]).

Processes statements SEQUENTIALLY with adaptive compute (1-8 internal ticks
per statement). This sequential processing is intentional — execution is sequential.

The negative eigenvalues enable state tracking that transformers cannot do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.components.state_slots import StateSlotBank
from sfm.components.deltanet_cell import DeltaNetStack
from sfm.components.adaptive_halting import AdaptiveHalting


class VariableBinder(nn.Module):
    """
    Binds input tokens to state slots via learned attention.

    Determines which slot should track which variable.
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        slot_dim: int
    ):
        """
        Initialize variable binder.

        Args:
            input_dim: Input embedding dimension.
            num_slots: Number of state slots.
            slot_dim: Dimension per slot.
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Query projection for binding
        self.query_proj = nn.Linear(input_dim, slot_dim)

        # Slot keys (learnable)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        # Binding confidence predictor
        self.confidence_net = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(),
            nn.Linear(slot_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute binding attention.

        Args:
            x: Input embeddings (batch, seq_len, input_dim).
            slots: Current slot states (batch, num_slots, slot_dim).

        Returns:
            Tuple of (bindings, binding_attention, confidence).
            - bindings: (batch, seq_len, slot_dim) - slot-weighted combination
            - binding_attention: (batch, seq_len, num_slots) - attention weights
            - confidence: (batch, seq_len, 1) - binding confidence
        """
        batch_size, seq_len, _ = x.shape

        # Project input to query
        queries = self.query_proj(x)  # (batch, seq_len, slot_dim)

        # Compute attention over slots
        # keys: (num_slots, slot_dim) -> broadcast to batch
        keys = self.slot_keys.unsqueeze(0).expand(batch_size, -1, -1)

        # Scaled dot-product attention
        scale = self.slot_dim ** 0.5
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale
        # (batch, seq_len, num_slots)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute binding (weighted combination of slots)
        bindings = torch.bmm(attn_weights, slots)  # (batch, seq_len, slot_dim)

        # Compute binding confidence
        confidence = self.confidence_net(bindings)  # (batch, seq_len, 1)

        return bindings, attn_weights, confidence


class StatementProcessor(nn.Module):
    """
    Processes a single statement with adaptive compute.

    Uses 1-8 internal ticks per statement depending on complexity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_slots: int,
        slot_dim: int,
        max_ticks: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize statement processor.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension for processing.
            num_slots: Number of state slots.
            slot_dim: Dimension per slot.
            max_ticks: Maximum internal ticks.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_ticks = max_ticks

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Variable binder
        self.binder = VariableBinder(hidden_dim, num_slots, slot_dim)

        # DeltaNet stack for sequential processing
        self.deltanet = DeltaNetStack(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout
        )

        # Adaptive halting
        self.halting = AdaptiveHalting(
            hidden_dim=hidden_dim,
            max_steps=max_ticks,
            halting_threshold=0.5
        )

        # Binding projection (slot_dim -> hidden_dim)
        self.binding_proj = nn.Linear(slot_dim, hidden_dim) if slot_dim != hidden_dim else nn.Identity()

        # Slot update projection (hidden_dim -> slot_dim)
        self.slot_update_proj = nn.Linear(hidden_dim, slot_dim) if slot_dim != hidden_dim else nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Layer norms
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        statement: torch.Tensor,
        slots: torch.Tensor,
        deltanet_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process a single statement with adaptive ticks.

        Args:
            statement: Statement embedding (batch, statement_len, input_dim).
            slots: Current slot states (batch, num_slots, slot_dim).
            deltanet_state: Previous DeltaNet state.

        Returns:
            Tuple of (output, updated_slots, updated_deltanet_state, info).
        """
        batch_size = statement.size(0)

        # Project input
        x = self.input_proj(statement)  # (batch, statement_len, hidden_dim)

        # Bind to slots
        bindings, binding_attn, confidence = self.binder(x, slots)

        # Combine with bindings (use binding_proj)
        bindings_projected = self.binding_proj(bindings)
        x = x + bindings_projected

        # Process through DeltaNet
        x, deltanet_state = self.deltanet(x, deltanet_state)

        # Aggregate statement representation
        # Use confidence-weighted mean
        statement_repr = (x * confidence).sum(dim=1) / (confidence.sum(dim=1) + 1e-6)
        # (batch, hidden_dim)

        # Adaptive halting
        final_repr, halt_info = self.halting(statement_repr)

        # Update slots based on binding attention and final representation
        # Each slot gets updated based on how much attention it received
        slot_update = torch.bmm(
            binding_attn.transpose(1, 2),  # (batch, num_slots, statement_len)
            x  # (batch, statement_len, hidden_dim)
        )  # (batch, num_slots, hidden_dim)

        # Project slot update to slot_dim
        slot_update_proj = self.slot_update_proj(slot_update)

        # Soft update to slots
        update_gate = torch.sigmoid(binding_attn.mean(dim=1, keepdim=True).transpose(1, 2))
        updated_slots = (1 - update_gate) * slots + update_gate * slot_update_proj

        # Output projection
        output = self.output_proj(x)

        info = {
            "binding_attention": binding_attn,
            "confidence": confidence,
            "ticks_used": halt_info["num_steps"],
            "ponder_cost": halt_info["ponder_cost"]
        }

        return output, updated_slots, deltanet_state, info


class ExecutionSystem(nn.Module):
    """
    System 2: Execution

    State Slot Bank with 64 registers that explicitly bind to variables
    and track values through execution.

    Key features:
    - Sequential processing (intentional - execution is sequential)
    - DeltaNet with eigenvalues in [-1, 1] for state tracking
    - Adaptive compute (1-8 ticks per statement)
    - Explicit variable binding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_slots: int = 64,
        slot_dim: int = 128,
        max_ticks: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize execution system.

        Args:
            input_dim: Input dimension from perception.
            hidden_dim: Internal hidden dimension.
            num_slots: Number of state slots (64 in spec).
            slot_dim: Dimension per slot.
            max_ticks: Maximum internal ticks per statement.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # State Slot Bank
        self.slot_bank = StateSlotBank(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_heads=num_heads,
            max_ticks=max_ticks,
            halting_threshold=0.5,
            dropout=dropout
        )

        # Statement processor
        self.processor = StatementProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            max_ticks=max_ticks,
            num_heads=num_heads,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout)
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize execution state.

        Args:
            batch_size: Batch size.
            device: Device.
            dtype: Data type.

        Returns:
            State dict with slots and deltanet state.
        """
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
        Forward pass through execution system.

        Processes input SEQUENTIALLY to track state through execution.

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

        # Process tokens as statements (group into chunks)
        # For simplicity, treat each token as a "mini-statement"
        # In production, this would be statement-level segmentation

        outputs = []
        total_ticks = 0

        for t in range(seq_len):
            token = x[:, t:t+1, :]  # (batch, 1, input_dim)

            # Process through slot bank
            slot_out, slots, slot_info = self.slot_bank(token, slots)

            # Process through statement processor
            proc_out, slots, deltanet_state, proc_info = self.processor(
                token, slots, deltanet_state
            )

            # Combine outputs
            combined = slot_out + proc_out
            outputs.append(combined)

            total_ticks += proc_info["ticks_used"]

        # Stack outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, input_dim)

        # Final projection
        output = self.output_proj(output)

        new_state = {
            "slots": slots,
            "deltanet_state": deltanet_state
        }

        info = {
            "avg_ticks_per_token": total_ticks / seq_len
        }

        if return_state:
            return output, new_state, info
        return output

    def get_slot_values(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get current slot values for inspection.

        Args:
            state: Execution state.

        Returns:
            Slot values (batch, num_slots, slot_dim).
        """
        return state["slots"]

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Execution System Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    input_dim = 128
    hidden_dim = 256
    num_slots = 32  # Reduced for testing
    slot_dim = 64
    max_ticks = 4  # Reduced for testing

    # Initialize execution system
    print("\n1. Initializing ExecutionSystem...")
    execution = ExecutionSystem(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_slots=num_slots,
        slot_dim=slot_dim,
        max_ticks=max_ticks,
        num_heads=4
    )

    # Test variable binder
    print("\n2. Testing VariableBinder...")
    binder = execution.processor.binder
    x = torch.randn(batch_size, seq_len, input_dim)
    slots = execution.slot_bank.init_slots(batch_size, x.device, x.dtype)

    bindings, attn, confidence = binder(x, slots)
    print(f"   Input shape: {x.shape}")
    print(f"   Bindings shape: {bindings.shape}")
    print(f"   Attention shape: {attn.shape}")
    print(f"   Confidence shape: {confidence.shape}")
    assert bindings.shape == (batch_size, seq_len, slot_dim), "Bindings shape mismatch!"
    assert attn.shape == (batch_size, seq_len, num_slots), "Attention shape mismatch!"
    print("   ✓ VariableBinder test passed!")

    # Test statement processor
    print("\n3. Testing StatementProcessor...")
    statement = torch.randn(batch_size, 4, input_dim)  # 4 tokens per statement
    output, new_slots, deltanet_state, info = execution.processor(statement, slots)
    print(f"   Statement shape: {statement.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Ticks used: {info['ticks_used']}")
    print(f"   Ponder cost: {info['ponder_cost']:.3f}")
    assert output.shape == (batch_size, 4, input_dim), "Processor output shape mismatch!"
    print("   ✓ StatementProcessor test passed!")

    # Test full execution system
    print("\n4. Testing full ExecutionSystem...")
    x = torch.randn(batch_size, seq_len, input_dim)
    output, state, info = execution.forward(x, return_state=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Average ticks per token: {info['avg_ticks_per_token']:.2f}")
    assert output.shape == x.shape, "Execution output shape mismatch!"
    assert "slots" in state, "State missing slots!"
    assert "deltanet_state" in state, "State missing deltanet_state!"
    print("   ✓ ExecutionSystem test passed!")

    # Test state persistence
    print("\n5. Testing state persistence...")
    x2 = torch.randn(batch_size, seq_len, input_dim)
    output2, state2, info2 = execution.forward(x2, state=state, return_state=True)
    print(f"   Second pass output shape: {output2.shape}")
    assert output2.shape == x2.shape, "Second output shape mismatch!"
    print("   ✓ State persistence test passed!")

    # Test slot inspection
    print("\n6. Testing slot inspection...")
    slot_values = execution.get_slot_values(state2)
    print(f"   Slot values shape: {slot_values.shape}")
    assert slot_values.shape == (batch_size, num_slots, slot_dim), "Slot values shape mismatch!"
    print("   ✓ Slot inspection test passed!")

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
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = execution.count_parameters()
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All Execution System tests passed!")
    print("=" * 60)
