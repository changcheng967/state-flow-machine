"""
State Slot Bank

64 registers that explicitly bind to variables and track values through execution.
This is the core of System 2 (Execution).

Each slot can:
1. Bind to a variable (learned soft attention over slots)
2. Store a value representation
3. Track mutations through sequential updates
4. Be queried for current state

The slot bank enables explicit state tracking that transformers cannot do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .deltanet_cell import DeltaNetCell


class StateSlotBank(nn.Module):
    """
    State Slot Bank: A memory bank of registers for tracking program state.

    Each slot can store a representation of a variable/value and be updated
    as the program executes. Slots are accessed via learned attention, allowing
    soft binding to variables.

    Architecture:
    - num_slots registers, each of dimension slot_dim
    - DeltaNet cell for sequential updates with negative eigenvalues
    - Attention-based read/write mechanism
    - Adaptive compute (can process multiple ticks per statement)
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int = 64,
        slot_dim: int = 128,
        num_heads: int = 4,
        max_ticks: int = 8,
        halting_threshold: float = 0.5,
        dropout: float = 0.1
    ):
        """
        Initialize State Slot Bank.

        Args:
            input_dim: Dimension of input embeddings.
            num_slots: Number of memory slots (default 64).
            slot_dim: Dimension of each slot.
            num_heads: Number of attention heads for slot access.
            max_ticks: Maximum internal ticks per input (adaptive halting).
            halting_threshold: Threshold for halting decision.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.max_ticks = max_ticks
        self.halting_threshold = halting_threshold

        # Slot memory: stores current state of all slots
        self.slot_memory = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.normal_(self.slot_memory, std=0.02)

        # Input projection to slot dimension
        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Query projection for attention over slots
        self.query_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.key_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.value_proj = nn.Linear(slot_dim, slot_dim, bias=False)

        # DeltaNet cell for sequential updates
        self.deltanet = DeltaNetCell(
            input_dim=slot_dim,
            hidden_dim=slot_dim,
            num_heads=num_heads,
            eigenvalue_init=0.9,
            dropout=dropout
        )

        # Write gate: controls how much to update each slot
        self.write_gate = nn.Linear(slot_dim, num_slots, bias=True)

        # Read gate: controls attention over slots for output
        self.read_gate = nn.Linear(slot_dim, num_heads, bias=True)

        # Halting network: decides when to stop processing
        self.halting_net = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(),
            nn.Linear(slot_dim // 2, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Linear(slot_dim, input_dim, bias=False)

        # Layer norms
        self.layer_norm_slots = nn.LayerNorm(slot_dim)
        self.layer_norm_input = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def init_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Initialize slot memory for a batch.

        Args:
            batch_size: Batch size.
            device: Target device.
            dtype: Data type.

        Returns:
            Initialized slot memory (batch, num_slots, slot_dim).
        """
        return self.slot_memory.expand(batch_size, -1, -1).clone()

    def read_from_slots(
        self,
        slots: torch.Tensor,
        query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from slots using attention.

        Args:
            slots: Slot memory (batch, num_slots, slot_dim).
            query: Query vector (batch, slot_dim).

        Returns:
            Tuple of (read_output, attention_weights).
        """
        batch_size = slots.size(0)

        # Project query and keys
        q = self.query_proj(query).unsqueeze(1)  # (batch, 1, slot_dim)
        k = self.key_proj(slots)  # (batch, num_slots, slot_dim)
        v = self.value_proj(slots)  # (batch, num_slots, slot_dim)

        # Scaled dot-product attention
        scale = self.slot_dim ** 0.5
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / scale  # (batch, 1, num_slots)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, num_slots)

        # Weighted sum of values
        read_output = torch.bmm(attn_weights, v).squeeze(1)  # (batch, slot_dim)
        attn_weights = attn_weights.squeeze(1)  # (batch, num_slots)

        return read_output, attn_weights

    def write_to_slots(
        self,
        slots: torch.Tensor,
        value: torch.Tensor,
        gate_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Write value to slots using learned gates.

        Args:
            slots: Current slot memory (batch, num_slots, slot_dim).
            value: Value to write (batch, slot_dim).
            gate_values: Optional pre-computed gate values (batch, num_slots).

        Returns:
            Updated slot memory (batch, num_slots, slot_dim).
        """
        batch_size = slots.size(0)

        if gate_values is None:
            gate_values = torch.sigmoid(self.write_gate(value))  # (batch, num_slots)

        # Compute DeltaNet update
        # Reshape for DeltaNet: treat each slot as a sequence
        value_expanded = value.unsqueeze(1).expand(-1, self.num_slots, -1)  # (batch, num_slots, slot_dim)

        # Process through DeltaNet for each slot
        updated_slots = []
        for i in range(self.num_slots):
            slot_input = value_expanded[:, i, :]  # (batch, slot_dim)
            slot_state = slots[:, i, :]  # (batch, slot_dim)
            updated, _ = self.deltanet(slot_input, slot_state)
            updated_slots.append(updated)

        updated_slots = torch.stack(updated_slots, dim=1)  # (batch, num_slots, slot_dim)

        # Apply gates
        gate_expanded = gate_values.unsqueeze(-1)  # (batch, num_slots, 1)
        new_slots = (1 - gate_expanded) * slots + gate_expanded * updated_slots

        return new_slots

    def compute_halting(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probability.

        Args:
            state: Current state (batch, slot_dim).

        Returns:
            Halting probability (batch, 1).
        """
        return self.halting_net(state)

    def forward(
        self,
        x: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through State Slot Bank with adaptive halting.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            slots: Optional pre-initialized slots (batch, num_slots, slot_dim).
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (output, new_slots, aux_info).
            - output: (batch, seq_len, input_dim)
            - new_slots: (batch, num_slots, slot_dim)
            - aux_info: Dict with auxiliary information
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize slots if needed
        if slots is None:
            slots = self.init_slots(batch_size, device, dtype)

        # Process sequence
        outputs = []
        all_attention = []
        total_ticks = 0

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            x_proj = self.input_proj(self.layer_norm_input(x_t))  # (batch, slot_dim)

            # Adaptive halting: process until halting threshold reached
            accumulated_output = torch.zeros(batch_size, self.slot_dim, device=device, dtype=dtype)
            remaining_prob = torch.ones(batch_size, 1, device=device, dtype=dtype)
            ticks_used = 0

            for tick in range(self.max_ticks):
                # Read from slots
                read_out, attn_weights = self.read_from_slots(slots, x_proj)
                accumulated_output = accumulated_output + remaining_prob * read_out

                # Compute halting
                halt_prob = self.compute_halting(read_out)

                # Write to slots
                write_gate = torch.sigmoid(self.write_gate(read_out))
                slots = self.write_to_slots(slots, read_out, write_gate)

                # Update remaining probability
                if tick < self.max_ticks - 1:
                    remaining_prob = remaining_prob * (1 - halt_prob)
                    ticks_used += 1

                # Check if we should halt
                if halt_prob.mean() > self.halting_threshold and tick > 0:
                    break

            total_ticks += ticks_used + 1

            # Final output projection
            output_t = self.output_proj(accumulated_output)
            outputs.append(output_t)

            if return_attention:
                all_attention.append(attn_weights)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, input_dim)

        aux_info = {
            "avg_ticks": total_ticks / seq_len,
            "attention_weights": all_attention if return_attention else None
        }

        return output, slots, aux_info


class StateSlotLayer(nn.Module):
    """
    Single layer wrapping StateSlotBank with residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        num_slots: int = 64,
        slot_dim: int = 128,
        num_heads: int = 4,
        max_ticks: int = 8,
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
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            slots: Optional slot memory (batch, num_slots, slot_dim).

        Returns:
            Tuple of (output, new_slots, aux_info).
        """
        residual = x
        x_norm = self.layer_norm(x)
        output, slots, aux_info = self.slot_bank(x_norm, slots)
        output = output + residual
        return output, slots, aux_info


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("State Slot Bank Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    input_dim = 64
    num_slots = 32  # Reduced for testing
    slot_dim = 128

    # Initialize slot bank
    print("\n1. Initializing StateSlotBank...")
    slot_bank = StateSlotBank(
        input_dim=input_dim,
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_heads=4,
        max_ticks=4,  # Reduced for testing
        halting_threshold=0.5
    )

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(batch_size, seq_len, input_dim)
    output, slots, aux_info = slot_bank(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Slots shape: {slots.shape}")
    print(f"   Average ticks per step: {aux_info['avg_ticks']:.2f}")

    assert output.shape == (batch_size, seq_len, input_dim), "Output shape mismatch!"
    assert slots.shape == (batch_size, num_slots, slot_dim), "Slots shape mismatch!"
    print("   ✓ Forward pass test passed!")

    # Test read operation
    print("\n3. Testing read operation...")
    query = torch.randn(batch_size, slot_dim)
    read_out, attn_weights = slot_bank.read_from_slots(slots, query)
    print(f"   Read output shape: {read_out.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert read_out.shape == (batch_size, slot_dim), "Read output shape mismatch!"
    assert attn_weights.shape == (batch_size, num_slots), "Attention weights shape mismatch!"
    print("   ✓ Read operation test passed!")

    # Test write operation
    print("\n4. Testing write operation...")
    value = torch.randn(batch_size, slot_dim)
    new_slots = slot_bank.write_to_slots(slots, value)
    print(f"   Updated slots shape: {new_slots.shape}")
    assert new_slots.shape == (batch_size, num_slots, slot_dim), "Write output shape mismatch!"
    print("   ✓ Write operation test passed!")

    # Test adaptive halting
    print("\n5. Testing adaptive halting...")
    state = torch.randn(batch_size, slot_dim)
    halt_prob = slot_bank.compute_halting(state)
    print(f"   Halting probability shape: {halt_prob.shape}")
    print(f"   Halting probability range: [{halt_prob.min():.3f}, {halt_prob.max():.3f}]")
    assert halt_prob.shape == (batch_size, 1), "Halting prob shape mismatch!"
    assert (halt_prob >= 0).all() and (halt_prob <= 1).all(), "Halting prob out of bounds!"
    print("   ✓ Adaptive halting test passed!")

    # Test layer
    print("\n6. Testing StateSlotLayer...")
    layer = StateSlotLayer(input_dim, num_slots, slot_dim)
    output, slots, aux_info = layer(x)
    print(f"   Layer output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, input_dim), "Layer output shape mismatch!"
    print("   ✓ Layer test passed!")

    # Test gradient flow
    print("\n7. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for name, param in slot_bank.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in slot_bank.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All State Slot Bank tests passed!")
    print("=" * 60)
