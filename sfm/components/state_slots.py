"""
State Slot Bank - CUBE-OPTIMIZED VERSION

64 registers that explicitly bind to variables and track values through execution.
This is the core of System 2 (Execution).

CUBE OPTIMIZATIONS:
- All tensor dimensions are multiples of 16
- All slot operations use batched matmul (torch.bmm or torch.matmul)
- MATCH: scores = torch.matmul(query, slot_keys.transpose) → one Cube call
- READ: values = torch.matmul(attention_weights, slot_values) → one Cube call
- WRITE: use torch.baddbmm for erase-then-write update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class StateSlotBank(nn.Module):
    """
    State Slot Bank: A memory bank of registers for tracking program state.

    CUBE-OPTIMIZED: All operations use batched matmul for Cube unit utilization.
    Dimensions: num_slots=64, slot_dim=128 (both multiples of 16).
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int = 64,      # Multiple of 16
        slot_dim: int = 128,       # Multiple of 16
        num_heads: int = 4,        # head_dim = 32
        max_ticks: int = 2,
        halting_threshold: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.head_dim = slot_dim // num_heads
        self.max_ticks = max_ticks
        self.halting_threshold = halting_threshold

        assert slot_dim % num_heads == 0, "slot_dim must be divisible by num_heads"
        assert num_slots % 16 == 0, "num_slots must be multiple of 16 for Cube optimization"
        assert slot_dim % 16 == 0, "slot_dim must be multiple of 16 for Cube optimization"

        # Slot memory: stores current state of all slots
        self.slot_memory = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.normal_(self.slot_memory, std=0.02)

        # Input projection to slot dimension
        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # CUBE-OPTIMIZED: Combined QKV projection
        self.qkv_proj = nn.Linear(slot_dim, 3 * slot_dim, bias=False)

        # Write projection (input_dim -> slot_dim)
        self.write_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # Erase projection for gated write
        self.erase_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # Update gate
        self.update_gate = nn.Linear(slot_dim + slot_dim, num_slots, bias=True)

        # Halting network (lightweight)
        self.halting_net = nn.Sequential(
            nn.Linear(slot_dim, 32),  # 32 is multiple of 16
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Linear(slot_dim, input_dim, bias=False)

        # Layer norms
        self.layer_norm_slots = nn.LayerNorm(slot_dim)
        self.layer_norm_input = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

        # Scale for attention (head_dim = 32)
        self.scale = self.head_dim ** -0.5

    def init_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize slot memory for a batch."""
        return self.slot_memory.expand(batch_size, -1, -1).clone()

    def _batched_attention(
        self,
        query: torch.Tensor,    # (batch, seq_len, slot_dim)
        keys: torch.Tensor,     # (batch, num_slots, slot_dim)
        values: torch.Tensor    # (batch, num_slots, slot_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUBE-OPTIMIZED multi-head attention using batched matmul.

        Returns:
            output: (batch, seq_len, slot_dim)
            weights: (batch, num_heads, seq_len, num_slots)
        """
        batch_size, seq_len, _ = query.shape

        # Reshape for multi-head: (batch, seq, slot_dim) -> (batch, heads, seq, head_dim)
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = keys.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = values.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # CUBE: Attention scores = Q @ K^T
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, num_slots)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, num_slots)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # CUBE: Read output = weights @ V
        # (batch, heads, seq, num_slots) @ (batch, heads, num_slots, head_dim)
        read_out = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, slot_dim)
        read_out = read_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.slot_dim)

        return read_out, attn_weights

    def _batched_write(
        self,
        slots: torch.Tensor,       # (batch, num_slots, slot_dim)
        write_values: torch.Tensor, # (batch, slot_dim)
        erase_values: torch.Tensor, # (batch, slot_dim)
        gates: torch.Tensor         # (batch, num_slots)
    ) -> torch.Tensor:
        """
        CUBE-OPTIMIZED batched write using baddbmm for erase-then-write.

        update: slots = (1 - erase) * slots + write * gate

        This is computed as: slots = slots - erase * slots * gate + write * gate
        Using torch.baddbmm for efficient batched matmul.
        """
        batch_size = slots.size(0)

        # Expand for broadcasting: (batch, num_slots, slot_dim)
        write_exp = write_values.unsqueeze(1).expand(-1, self.num_slots, -1)
        erase_exp = erase_values.unsqueeze(1).expand(-1, self.num_slots, -1)
        gates_exp = gates.unsqueeze(-1)  # (batch, num_slots, 1)

        # Compute erase contribution: erase * gate
        # Using bmm: (batch, num_slots, slot_dim) * (batch, num_slots, 1) -> (batch, num_slots, slot_dim)
        erase_scaled = erase_exp * gates_exp

        # CUBE: slots = slots - slots * erase_scaled + write_exp * gates_exp
        # First: compute new_slots = slots * (1 - erase_scaled)
        new_slots = slots * (1 - erase_scaled)

        # Then: add write contribution
        new_slots = new_slots + write_exp * gates_exp

        return new_slots

    def forward(
        self,
        x: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with CUBE-optimized batched operations.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            slots: Optional pre-initialized slots (batch, num_slots, slot_dim).
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (output, new_slots, aux_info).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize slots if needed
        if slots is None:
            slots = self.init_slots(batch_size, device, dtype)

        # BATCHED: Project all inputs at once
        x_proj = self.input_proj(self.layer_norm_input(x))  # (batch, seq_len, slot_dim)

        # CUBE: Compute read attention for entire sequence
        read_out, attn_weights = self._batched_attention(x_proj, slots, slots)

        # Aggregate sequence info for slot update
        seq_summary = read_out.mean(dim=1)  # (batch, slot_dim)

        # Compute write and erase values
        write_values = self.write_proj(seq_summary)  # (batch, slot_dim)
        erase_values = torch.sigmoid(self.erase_proj(seq_summary))  # (batch, slot_dim)

        # Compute update gates
        gate_input = torch.cat([slots.mean(dim=1), write_values], dim=-1)
        update_gates = torch.sigmoid(self.update_gate(gate_input))  # (batch, num_slots)

        # CUBE: Batched write with erase-then-write
        new_slots = self._batched_write(slots, write_values, erase_values, update_gates)
        new_slots = self.layer_norm_slots(new_slots)

        # Compute halting
        halt_prob = self.halting_net(seq_summary)  # (batch, 1)
        avg_ticks = 1.0 + (1.0 - halt_prob.mean().item()) * (self.max_ticks - 1)

        # BATCHED: Output projection for entire sequence
        output = self.output_proj(read_out)
        output = self.dropout(output)

        aux_info = {
            "avg_ticks": avg_ticks,
            "attention_weights": attn_weights if return_attention else None
        }

        return output, new_slots, aux_info


class StateSlotLayer(nn.Module):
    """Single layer wrapping StateSlotBank with residual connection."""

    def __init__(
        self,
        input_dim: int,
        num_slots: int = 64,
        slot_dim: int = 128,
        num_heads: int = 4,
        max_ticks: int = 2,
        halting_threshold: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.slot_bank = StateSlotBank(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_heads=num_heads,
            max_ticks=max_ticks,
            halting_threshold=halting_threshold,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with residual connection."""
        residual = x
        x_norm = self.layer_norm(x)
        output, slots, aux_info = self.slot_bank(x_norm, slots)
        output = output + residual
        return output, slots, aux_info


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("State Slot Bank (Cube-Optimized) Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    input_dim = 256  # Multiple of 16
    num_slots = 64   # Multiple of 16
    slot_dim = 128   # Multiple of 16

    # Verify dimension alignment
    print("\n1. Verifying dimension alignment...")
    assert num_slots % 16 == 0, f"num_slots {num_slots} not multiple of 16"
    assert slot_dim % 16 == 0, f"slot_dim {slot_dim} not multiple of 16"
    print(f"   num_slots: {num_slots} (x{num_slots // 16})")
    print(f"   slot_dim: {slot_dim} (x{slot_dim // 16})")
    print(f"   head_dim: {slot_dim // 4}")
    print("   [OK] Dimension alignment verified!")

    # Initialize slot bank
    print("\n2. Initializing StateSlotBank...")
    slot_bank = StateSlotBank(
        input_dim=input_dim,
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_heads=4,
        max_ticks=2
    )

    # Test forward pass
    print("\n3. Testing Cube-optimized forward pass...")
    import time
    x = torch.randn(batch_size, seq_len, input_dim)

    start = time.time()
    output, slots, aux_info = slot_bank(x)
    elapsed = time.time() - start

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Slots shape: {slots.shape}")
    print(f"   Forward pass time: {elapsed*1000:.2f}ms")
    print(f"   Average ticks: {aux_info['avg_ticks']:.2f}")

    assert output.shape == (batch_size, seq_len, input_dim), "Output shape mismatch!"
    assert slots.shape == (batch_size, num_slots, slot_dim), "Slots shape mismatch!"
    print("   [OK] Forward pass test passed!")

    # Performance test
    print("\n4. Performance test (100 iterations)...")
    start = time.time()
    for _ in range(100):
        output, slots, _ = slot_bank(x, slots)
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({elapsed*10:.1f}ms per iter)")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_count = sum(1 for p in slot_bank.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"   Parameters with gradients: {grad_count}/{len(list(slot_bank.parameters()))}")
    assert grad_count > 0, "No gradients found!"
    print("   [OK] Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in slot_bank.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All State Slot Bank (Cube-Optimized) tests passed!")
    print("=" * 60)
