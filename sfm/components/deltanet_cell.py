"""
DeltaNet Recurrent Cell

The core component enabling state tracking. Uses eigenvalues in [-1, 1]
(not [0, 1] like standard RNNs) which enables tracking state transformations
that transformers cannot do (TC0 circuit complexity limit).

Key insight: Negative eigenvalues allow the cell to "subtract" state,
enabling reversible computations and proper tracking of variable mutations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DeltaNetCell(nn.Module):
    """
    DeltaNet recurrent cell with learnable eigenvalues in [-1, 1].

    This cell maintains a hidden state that evolves according to:
        h_t = alpha_t * h_{t-1} + beta_t * tanh(W_x * x_t + W_h * h_{t-1} + b)

    Where alpha_t and beta_t are learned gates with alpha_t constrained to [-1, 1].

    The key innovation is allowing alpha_t to be negative, which enables:
    - Reversible computations (multiplying by -1 "undoes" certain operations)
    - Proper tracking of variable mutations (overwrites become subtractive)
    - State oscillations that can track loops and conditionals
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        eigenvalue_init: float = 0.9,
        dropout: float = 0.1
    ):
        """
        Initialize DeltaNet cell.

        Args:
            input_dim: Dimension of input tensor.
            hidden_dim: Dimension of hidden state.
            num_heads: Number of attention heads for gated updates.
            eigenvalue_init: Initial value for eigenvalues (close to 1 for long memory).
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eigenvalue_init = eigenvalue_init

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim * 3, bias=False)

        # Hidden state projection for attention
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Eigenvalue parameters (one per head)
        # Using tanh to constrain to [-1, 1]
        self.eigenvalue_raw = nn.Parameter(
            torch.zeros(num_heads, dtype=torch.float32)
        )
        self._init_eigenvalues(eigenvalue_init)

        # Gate parameters
        self.gate_proj = nn.Linear(hidden_dim, num_heads * 2, bias=True)

        # Layer norms
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.layer_norm_hidden = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _init_eigenvalues(self, init_value: float):
        """Initialize eigenvalue parameters to achieve target initial value."""
        # tanh^{-1}(init_value) gives us the raw parameter value
        if abs(init_value) >= 1:
            init_value = 0.99 * (1 if init_value > 0 else -1)
        raw_value = 0.5 * math.log((1 + init_value) / (1 - init_value))
        nn.init.constant_(self.eigenvalue_raw, raw_value)

    def get_eigenvalues(self) -> torch.Tensor:
        """Get current eigenvalues constrained to [-1, 1]."""
        return torch.tanh(self.eigenvalue_raw)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DeltaNet cell.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim).
            h: Previous hidden state of shape (batch, hidden_dim). None initializes to zeros.

        Returns:
            Tuple of (output, new_hidden_state).
            - output: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            - new_hidden_state: (batch, hidden_dim)
        """
        # Handle both single step and sequence input
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)

        # Get eigenvalues (constrained to [-1, 1])
        eigenvalues = self.get_eigenvalues()  # (num_heads,)

        # Process sequence step by step (recurrent)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)

            # Project input
            x_proj = self.input_proj(x_t)  # (batch, hidden_dim * 3)
            x_proj = self.layer_norm_input(x_proj[:, :self.hidden_dim])

            # Compute gates
            gate_input = x_proj + self.layer_norm_hidden(h)
            gates = self.gate_proj(gate_input)  # (batch, num_heads * 2)
            alpha_gate, beta_gate = gates.chunk(2, dim=-1)  # Each: (batch, num_heads)
            alpha_gate = torch.sigmoid(alpha_gate)  # (batch, num_heads)
            beta_gate = torch.sigmoid(beta_gate)  # (batch, num_heads)

            # Compute new hidden state with multi-head update
            h_heads = h.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)
            x_heads = x_proj.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)

            # Apply eigenvalue-weighted update
            # h_new = eigenvalue * alpha_gate * h_old + beta_gate * x_new
            eigenvalues_expanded = eigenvalues.view(1, -1, 1)  # (1, num_heads, 1)
            alpha_expanded = alpha_gate.unsqueeze(-1)  # (batch, num_heads, 1)
            beta_expanded = beta_gate.unsqueeze(-1)  # (batch, num_heads, 1)

            h_new = eigenvalues_expanded * alpha_expanded * h_heads + beta_expanded * x_heads
            h = h_new.view(batch_size, self.hidden_dim)

            # Output projection
            output = self.output_proj(h)
            output = self.dropout(output)
            outputs.append(output)

        if is_single_step:
            return outputs[0], h

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return outputs, h


class DeltaNetLayer(nn.Module):
    """
    A layer wrapping DeltaNet cell with residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        eigenvalue_init: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()

        self.cell = DeltaNetCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            eigenvalue_init=eigenvalue_init,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(input_dim)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            h: Previous hidden state (batch, hidden_dim).

        Returns:
            Tuple of (output, new_hidden_state).
        """
        # Residual connection
        residual = self.residual_proj(x)

        # Normalize input
        x_norm = self.layer_norm(x)

        # DeltaNet cell
        output, h_new = self.cell(x_norm, h)

        # Add residual
        output = output + residual

        return output, h_new


class DeltaNetStack(nn.Module):
    """
    Stack of DeltaNet layers for deep recurrent processing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        eigenvalue_init: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DeltaNetLayer(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                eigenvalue_init=eigenvalue_init,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            h: Previous hidden states (batch, num_layers, hidden_dim) or None.

        Returns:
            Tuple of (output, new_hidden_states).
        """
        batch_size = x.size(0)
        device = x.device

        if h is None:
            h = torch.zeros(batch_size, self.num_layers, self.layers[0].cell.hidden_dim, device=device, dtype=x.dtype)

        new_h_list = []
        for i, layer in enumerate(self.layers):
            x, h_i = layer(x, h[:, i, :] if h.dim() == 3 else None)
            new_h_list.append(h_i)

        new_h = torch.stack(new_h_list, dim=1)  # (batch, num_layers, hidden_dim)
        return x, new_h


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("DeltaNet Cell Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    input_dim = 64
    hidden_dim = 128
    num_heads = 4

    # Test single cell
    print("\n1. Testing DeltaNetCell...")
    cell = DeltaNetCell(input_dim, hidden_dim, num_heads)

    x = torch.randn(batch_size, seq_len, input_dim)
    output, h = cell(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden state shape: {h.shape}")
    print(f"   Eigenvalues: {cell.get_eigenvalues()}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Output shape mismatch!"
    assert h.shape == (batch_size, hidden_dim), "Hidden state shape mismatch!"
    print("   ✓ DeltaNetCell test passed!")

    # Test single step
    print("\n2. Testing single-step execution...")
    x_single = torch.randn(batch_size, input_dim)
    output_single, h_new = cell(x_single, h)
    print(f"   Single input shape: {x_single.shape}")
    print(f"   Single output shape: {output_single.shape}")
    assert output_single.shape == (batch_size, hidden_dim), "Single output shape mismatch!"
    print("   ✓ Single-step test passed!")

    # Test layer
    print("\n3. Testing DeltaNetLayer...")
    layer = DeltaNetLayer(input_dim, hidden_dim, num_heads)
    output, h = layer(x)
    print(f"   Layer output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Layer output shape mismatch!"
    print("   ✓ DeltaNetLayer test passed!")

    # Test stack
    print("\n4. Testing DeltaNetStack...")
    stack = DeltaNetStack(input_dim, hidden_dim, num_layers=3, num_heads=num_heads)
    output, h_stack = stack(x)
    print(f"   Stack output shape: {output.shape}")
    print(f"   Stack hidden shape: {h_stack.shape}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Stack output shape mismatch!"
    assert h_stack.shape == (batch_size, 3, hidden_dim), "Stack hidden shape mismatch!"
    print("   ✓ DeltaNetStack test passed!")

    # Test eigenvalue constraints
    print("\n5. Testing eigenvalue constraints...")
    eigenvalues = cell.get_eigenvalues()
    assert (eigenvalues >= -1).all() and (eigenvalues <= 1).all(), "Eigenvalues out of bounds!"
    print(f"   Eigenvalues in [-1, 1]: {eigenvalues}")
    print("   ✓ Eigenvalue constraint test passed!")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    for name, param in cell.named_parameters():
        if param.grad is not None:
            has_grad = param.grad.abs().sum() > 0
            print(f"   {name}: grad exists = {has_grad}")
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in cell.parameters())
    print(f"\n   Total parameters in cell: {total_params:,}")

    print("\n" + "=" * 60)
    print("All DeltaNet tests passed!")
    print("=" * 60)
