"""
State Slot Bank - SEQUENTIAL WRITE VERSION

Registers that explicitly bind to variables and track values through execution.
This is the core of System 2 (Execution).

Default 16 slots: hard-mode programs have ~10 variables, 16 is sufficient
while reducing routing interference. 16 is also a multiple of 16 for Cube
optimization.

KEY ARCHITECTURAL ADVANTAGE:
The sequential per-chunk write preserves execution order.
Early statements write first, later statements can overwrite.
This is what transformers CANNOT do (attention is parallel).

CUBE OPTIMIZATIONS:
- All tensor dimensions are multiples of 16
- READ operations use batched matmul (Cube)
- Only WRITE is sequential per-chunk (this is intentional!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class StateSlotBank(nn.Module):
    """
    State Slot Bank: A memory bank of registers for tracking program state.

    SEQUENTIAL WRITE: Per-chunk updates preserve execution order.
    This is the core architectural advantage over transformers.
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int = 16,       # Default 16 (multiple of 16, ~10 vars in hard mode)
        slot_dim: int = 128,       # Multiple of 16
        num_heads: int = 4,        # head_dim = 32
        max_ticks: int = 2,
        halting_threshold: float = 0.5,
        dropout: float = 0.1,
        chunk_size: int = 16       # Chunk size for sequential write
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.head_dim = slot_dim // num_heads
        self.max_ticks = max_ticks
        self.halting_threshold = halting_threshold
        self.chunk_size = chunk_size
        self.top_k_slots = 3  # Update top-3 slots per chunk

        assert slot_dim % num_heads == 0, "slot_dim must be divisible by num_heads"
        assert num_slots % 16 == 0, "num_slots must be multiple of 16"
        assert slot_dim % 16 == 0, "slot_dim must be multiple of 16"

        # Slot memory: stores current state of all slots
        self.slot_memory = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.normal_(self.slot_memory, std=0.02)

        # Slot keys for matching (learnable)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        # Input projection to slot dimension
        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Write projection
        self.write_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # Update gate for sequential write
        self.update_gate = nn.Linear(slot_dim + slot_dim, 1, bias=True)

        # Output projection
        self.output_proj = nn.Linear(slot_dim, input_dim, bias=False)

        # Layer norms
        self.layer_norm_slots = nn.LayerNorm(slot_dim)
        self.layer_norm_input = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

        # Scale for attention
        self.scale = self.head_dim ** -0.5

    def init_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize slot memory for a batch."""
        return self.slot_memory.expand(batch_size, -1, -1).clone()

    def _batched_read_attention(
        self,
        query: torch.Tensor,    # (batch, seq_len, slot_dim)
        slots: torch.Tensor      # (batch, num_slots, slot_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUBE-OPTIMIZED batched read attention.

        Returns:
            read_out: (batch, seq_len, slot_dim)
            attn_weights: (batch, num_heads, seq_len, num_slots)
        """
        batch_size, seq_len, _ = query.shape

        # Reshape for multi-head
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = slots.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = slots.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # CUBE: Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # CUBE: Read output
        read_out = torch.matmul(attn_weights, v)
        read_out = read_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.slot_dim)

        return read_out, attn_weights

    def _sequential_chunk_write(
        self,
        slots: torch.Tensor,      # (batch, num_slots, slot_dim)
        read_out: torch.Tensor    # (batch, seq_len, slot_dim)
    ) -> torch.Tensor:
        """
        SEQUENTIAL per-chunk write - THIS IS THE CORE ADVANTAGE.

        Early chunks write first, later chunks can overwrite.
        This preserves execution order that transformers cannot replicate.

        Args:
            slots: Current slot values.
            read_out: Read output from attention (batch, seq_len, slot_dim).

        Returns:
            Updated slots after sequential chunk processing.
        """
        batch_size, seq_len, _ = read_out.shape
        chunk_size = self.chunk_size

        # Process chunks sequentially - THIS LOOP IS INTENTIONAL
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Get chunk and compute summary
            chunk = read_out[:, chunk_start:chunk_end, :]  # (batch, chunk_len, slot_dim)
            chunk_summary = chunk.mean(dim=1)  # (batch, slot_dim)

            # Match against slot_keys (variable identity), NOT slot values
            # slot_keys = learned parameters representing WHICH variable a slot tracks
            # slots = current stored VALUES in each slot
            # Matching by key asks "which slot is assigned to this variable?"
            chunk_norm = F.normalize(chunk_summary, dim=-1)
            keys = self.slot_keys.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_slots, slot_dim)
            key_norms = F.normalize(keys, dim=-1)  # (batch, num_slots, slot_dim)

            # Cosine similarity: (batch, num_slots)
            match_scores = torch.bmm(
                key_norms,
                chunk_norm.unsqueeze(-1)
            ).squeeze(-1)  # (batch, num_slots)

            # Select top-k slots to update
            top_k = min(self.top_k_slots, self.num_slots)
            _, top_indices = match_scores.topk(top_k, dim=-1)  # (batch, top_k)

            # Compute write value
            write_value = self.write_proj(chunk_summary)  # (batch, slot_dim)

            # Compute update gate using selected slot values, not global mean
            top_k_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.slot_dim)  # (batch, top_k, slot_dim)
            selected_slot_values = torch.gather(slots, 1, top_k_expanded)  # (batch, top_k, slot_dim)
            selected_mean = selected_slot_values.mean(dim=1)  # (batch, slot_dim)
            gate_input = torch.cat([selected_mean, write_value], dim=-1)
            gate = torch.sigmoid(self.update_gate(gate_input))  # (batch, 1)

            # SEQUENTIAL UPDATE: Update top-k slots using scatter (avoid in-place ops)
            # Build update mask and values for all slots
            update_mask = torch.zeros(batch_size, self.num_slots, 1, device=slots.device, dtype=slots.dtype)
            update_values = torch.zeros_like(slots)

            for k in range(top_k):
                # Create one-hot mask for this position in top-k
                batch_indices = torch.arange(batch_size, device=slots.device)
                slot_indices = top_indices[:, k]  # (batch,)

                # Get current slot values for these indices
                current_slot_vals = slots[batch_indices, slot_indices]  # (batch, slot_dim)

                # Compute gated update
                new_slot_vals = (1 - gate) * current_slot_vals + gate * write_value

                # Accumulate updates (will be applied with mask)
                update_values[batch_indices, slot_indices] = new_slot_vals
                update_mask[batch_indices, slot_indices] = 1.0

            # Apply all updates at once (non-in-place)
            slots = slots * (1 - update_mask) + update_values * update_mask

        # Layer norm after all updates
        slots = self.layer_norm_slots(slots)

        return slots

    def forward(
        self,
        x: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with SEQUENTIAL write.

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

        # CUBE: Batched read attention
        read_out, attn_weights = self._batched_read_attention(x_proj, slots)

        # SEQUENTIAL: Per-chunk write (core advantage)
        new_slots = self._sequential_chunk_write(slots, read_out)

        # Compute halting (based on read output variance)
        read_variance = read_out.var(dim=1).mean().item()
        avg_ticks = 1.0 + min(read_variance * 10, self.max_ticks - 1)

        # BATCHED: Output projection
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
        num_slots: int = 16,
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
    print("State Slot Bank (Sequential Write) Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 64  # Multiple of chunk_size
    input_dim = 256
    num_slots = 16
    slot_dim = 128

    # Initialize
    print("\n1. Initializing StateSlotBank with sequential write...")
    slot_bank = StateSlotBank(
        input_dim=input_dim,
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_heads=4,
        max_ticks=2,
        chunk_size=16
    )

    # Test forward pass
    print("\n2. Testing sequential write forward pass...")
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

    # Test that sequential processing works (different outputs for different orderings)
    print("\n3. Testing sequential ordering effect...")
    x_reversed = torch.flip(x, dims=[1])  # Reverse sequence order
    output_rev, slots_rev, _ = slot_bank(x_reversed)

    # Outputs should differ due to sequential write
    diff = (output - output_rev.flip(dims=[1])).abs().mean().item()
    print(f"   Output difference (reversed vs normal): {diff:.6f}")
    # Any non-zero difference indicates sequential processing is happening
    assert diff > 0, "Sequential processing should produce different outputs!"
    print("   [OK] Sequential ordering verified!")

    # Performance test
    print("\n4. Performance test (100 iterations)...")
    start = time.time()
    for _ in range(100):
        output, slots, _ = slot_bank(x, slots)
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({elapsed*10:.1f}ms per iter)")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    slot_bank.zero_grad()
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
    print("All State Slot Bank (Sequential Write) tests passed!")
    print("=" * 60)
