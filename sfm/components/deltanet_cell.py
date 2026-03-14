"""
DeltaNet Recurrent Cell - CUBE-FIRST PARALLEL SCAN

The core component enabling state tracking. Uses eigenvalues in [-1, 1]
(not [0, 1] like standard RNNs) which enables tracking state transformations
that transformers cannot do (TC0 circuit complexity limit).

Key insight: Negative eigenvalues allow the cell to "subtract" state,
enabling reversible computations and proper tracking of variable mutations.

CUBE-FIRST OPTIMIZATION for Ascend NPU:
- Reshape sequence into chunks of size 16 (matches Cube unit width)
- Within-chunk recurrence uses MATRIX MULTIPLICATION (runs on Cube)
- Between-chunk carry uses Vector unit
- This gives Cube+Vector pipeline parallelism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DeltaNetCell(nn.Module):
    """
    DeltaNet recurrent cell with learnable eigenvalues in [-1, 1].

    CUBE-FIRST: Uses matrix-based parallel scan within 16-step chunks.
    The 16x16 lower-triangular matrix M is computed and multiplied,
    utilizing the DaVinci Cube unit efficiently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        eigenvalue_init: float = 0.9,
        dropout: float = 0.1,
        chunk_size: int = 16  # Cube unit width for optimal performance
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eigenvalue_init = eigenvalue_init
        self.chunk_size = chunk_size

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert hidden_dim % 16 == 0, "hidden_dim must be multiple of 16 for Cube optimization"
        assert self.head_dim % 16 == 0 or self.head_dim >= 16, "head_dim should be >= 16 for Cube"

        # Combined projections for efficiency
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, num_heads * 2, bias=True)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Eigenvalue parameters (one per head) - constrained to [-1, 1] via tanh
        self.eigenvalue_raw = nn.Parameter(torch.zeros(num_heads))
        self._init_eigenvalues(eigenvalue_init)

        # Layer norms
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Pre-compute chunk scan matrices for chunk_size=16
        self._register_chunk_matrices()

    def _init_eigenvalues(self, init_value: float):
        """Initialize eigenvalue parameters."""
        if abs(init_value) >= 1:
            init_value = 0.99 * (1 if init_value > 0 else -1)
        raw_value = 0.5 * math.log((1 + init_value) / (1 - init_value))
        nn.init.constant_(self.eigenvalue_raw, raw_value)

    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues constrained to [-1, 1]."""
        return torch.tanh(self.eigenvalue_raw)

    def _register_chunk_matrices(self):
        """Pre-compute indices for chunk-based parallel scan."""
        # For chunk_size=16, we create a lower triangular matrix pattern
        # This is used to compute the prefix products/sums efficiently
        chunk_size = self.chunk_size
        # Register indices for lower triangular construction
        tri_indices = torch.tril_indices(chunk_size, chunk_size)
        self.register_buffer('tri_indices', tri_indices, persistent=False)

    def _compute_chunk_matrix(self, a_chunk: torch.Tensor) -> torch.Tensor:
        """
        Compute the 16x16 lower-triangular matrix M for chunk scan.

        M[i,j] = prod(a_k, k=j+1..i) for i>j
        M[i,i] = 1
        M[i,j] = 0 for i<j

        This matrix multiply runs on the Cube unit: M @ B_chunk

        Args:
            a_chunk: (batch, chunk_len, num_heads) - decay factors

        Returns:
            M: (batch, chunk_len, chunk_len, num_heads) - lower triangular matrices
        """
        batch_size, chunk_len, num_heads = a_chunk.shape

        # Compute cumulative products: cumprod[i] = a_0 * a_1 * ... * a_i
        cumprod = torch.cumprod(a_chunk, dim=1)  # (batch, chunk_len, num_heads)

        # For M[i,j] = prod(a_k, k=j+1..i) = cumprod[i] / cumprod[j]
        # For i=j: M[i,i] = 1
        # For i<j: M[i,j] = 0

        # Expand for broadcasting: (batch, chunk_len, 1, num_heads) / (batch, 1, chunk_len, num_heads)
        cumprod_i = cumprod.unsqueeze(2)  # (batch, chunk_len, 1, num_heads)
        cumprod_j = cumprod.unsqueeze(1)  # (batch, 1, chunk_len, num_heads)

        # Compute ratio: (batch, chunk_len, chunk_len, num_heads)
        # Use safe division with 1.0 for positions where j > i (will be zeroed anyway)
        # M[i,j] = cumprod[i] / cumprod[j-1] for i >= j, with cumprod[-1] = 1

        # For i >= j: M[i,j] = cumprod[i] / (cumprod[j-1] if j > 0 else 1)
        # We can compute this as: cumprod[i] / cumprod[j] * a_j for i > j

        # Simpler approach: create lower triangular mask and use cumprod
        # M[i,j] = cumprod[i] / cumprod[j-1] for i > j (with cumprod[-1] = 1)
        # M[i,i] = 1

        # Create padded cumprod with 1 at position -1
        ones = torch.ones(batch_size, 1, num_heads, device=a_chunk.device, dtype=a_chunk.dtype)
        cumprod_padded = torch.cat([ones, cumprod], dim=1)  # (batch, chunk_len+1, num_heads)

        # M[i,j] = cumprod[i] / cumprod[j-1+1] = cumprod[i] / cumprod[j] for i >= j
        # But we need to handle the indexing correctly

        # Create lower triangular mask
        mask = torch.tril(torch.ones(chunk_len, chunk_len, device=a_chunk.device, dtype=a_chunk.dtype))
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, chunk_len, chunk_len, 1)

        # For positions i >= j: M[i,j] = cumprod[i] / cumprod_padded[j]
        # where cumprod_padded[j] = 1 for j=0, else cumprod[j-1]
        cumprod_j_padded = cumprod_padded[:, :-1, :].unsqueeze(1)  # (batch, 1, chunk_len, num_heads)
        cumprod_i_exp = cumprod.unsqueeze(2)  # (batch, chunk_len, 1, num_heads)

        # Safe division
        M = torch.where(
            cumprod_j_padded.abs() > 1e-8,
            cumprod_i_exp / cumprod_j_padded,
            torch.zeros_like(cumprod_i_exp)
        )

        # Apply mask and set diagonal to 1
        M = M * mask

        # Set diagonal to 1 (i=j positions)
        diag_mask = torch.eye(chunk_len, device=a_chunk.device, dtype=a_chunk.dtype)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1)  # (1, chunk_len, chunk_len, 1)
        M = M * (1 - diag_mask) + diag_mask

        return M  # (batch, chunk_len, chunk_len, num_heads)

    def _cube_parallel_scan(
        self,
        a: torch.Tensor,  # (batch, seq_len, num_heads, 1) - decay factors
        b: torch.Tensor,  # (batch, seq_len, num_heads, head_dim) - inputs
        h0: torch.Tensor  # (batch, num_heads, head_dim) - initial state
    ) -> torch.Tensor:
        """
        Cube-first parallel scan for linear recurrence: h_t = a_t * h_{t-1} + b_t

        Uses chunked matrix-based scan for within-chunk computation (Cube unit),
        with sequential carry between chunks (Vector unit).
        """
        batch_size, seq_len, num_heads, head_dim = b.shape
        device = b.device
        dtype = b.dtype

        chunk_size = self.chunk_size

        # Pad sequence to multiple of chunk_size
        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            a = F.pad(a, (0, 0, 0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, 0, 0, pad_len))
            padded_seq_len = seq_len + pad_len
        else:
            padded_seq_len = seq_len

        num_chunks = padded_seq_len // chunk_size

        # Reshape to chunks: (batch, num_chunks, chunk_size, ...)
        a_flat = a.squeeze(-1)  # (batch, padded_seq_len, num_heads)
        a_chunks = a_flat.view(batch_size, num_chunks, chunk_size, num_heads)
        b_chunks = b.view(batch_size, num_chunks, chunk_size, num_heads, head_dim)

        # Process each chunk
        h = h0  # (batch, num_heads, head_dim)
        outputs = []

        for chunk_idx in range(num_chunks):
            a_chunk = a_chunks[:, chunk_idx, :, :]  # (batch, chunk_size, num_heads)
            b_chunk = b_chunks[:, chunk_idx, :, :, :]  # (batch, chunk_size, num_heads, head_dim)

            # CUBE OPERATION: Compute chunk scan matrix M
            M = self._compute_chunk_matrix(a_chunk)  # (batch, chunk_size, chunk_size, num_heads)

            # CUBE OPERATION: M @ B_chunk - matrix multiply for each batch and head
            # Reshape for batched matmul: treat (batch, num_heads) as batch dims
            # M: (batch, chunk_size, chunk_size, num_heads) -> (batch*num_heads, chunk_size, chunk_size)
            # B: (batch, chunk_size, num_heads, head_dim) -> (batch*num_heads, chunk_size, head_dim)

            M_flat = M.permute(0, 3, 1, 2).reshape(batch_size * num_heads, chunk_size, chunk_size)
            B_flat = b_chunk.permute(0, 2, 3, 1).reshape(batch_size * num_heads, chunk_size, head_dim)

            # CUBE: Matrix multiply M @ B
            h_contrib = torch.bmm(M_flat, B_flat)  # (batch*num_heads, chunk_size, head_dim)
            h_contrib = h_contrib.view(batch_size, num_heads, chunk_size, head_dim)
            h_contrib = h_contrib.permute(0, 2, 1, 3)  # (batch, chunk_size, num_heads, head_dim)

            # VECTOR OPERATION: Add initial state contribution
            # h_chunk[t] includes h0 * (prod a_0..a_{t-1}) contribution
            # This is the first term of the scan
            cum_a = torch.cumprod(a_chunk, dim=1)  # (batch, chunk_size, num_heads)
            # h: (batch, num_heads, head_dim) -> (batch, 1, num_heads, head_dim)
            # cum_a: (batch, chunk_size, num_heads) -> (batch, chunk_size, num_heads, 1)
            h0_contrib = cum_a.unsqueeze(-1) * h.unsqueeze(1)  # (batch, chunk_size, num_heads, head_dim)

            # Total: h_chunk = h0_contrib + h_contrib
            h_chunk = h0_contrib + h_contrib  # (batch, chunk_size, num_heads, head_dim)

            outputs.append(h_chunk)

            # VECTOR OPERATION: Carry forward last hidden state for next chunk
            h = h_chunk[:, -1, :, :]  # (batch, num_heads, head_dim)

        # Concatenate all chunks
        h_all = torch.cat(outputs, dim=1)  # (batch, padded_seq_len, num_heads, head_dim)

        # Remove padding
        h_all = h_all[:, :seq_len, :, :]  # (batch, seq_len, num_heads, head_dim)

        return h_all

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Cube-first parallel scan.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim).
            h: Previous hidden state (batch, hidden_dim).

        Returns:
            Tuple of (output, new_hidden_state).
        """
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        # Get eigenvalues
        eigenvalues = self.get_eigenvalues()  # (num_heads,)

        # BATCHED: Project all inputs at once
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x_proj = self.layer_norm(x_proj)

        # BATCHED: Compute gates for all positions
        gates = self.gate_proj(x_proj)  # (batch, seq_len, num_heads * 2)
        alpha_gate, beta_gate = gates.chunk(2, dim=-1)  # Each: (batch, seq_len, num_heads)
        alpha_gate = torch.sigmoid(alpha_gate)
        beta_gate = torch.sigmoid(beta_gate)

        # Compute decay factors: a_t = eigenvalue * alpha_t
        eigenvalues_exp = eigenvalues.view(1, 1, self.num_heads, 1)  # (1, 1, num_heads, 1)
        alpha_exp = alpha_gate.unsqueeze(-1)  # (batch, seq_len, num_heads, 1)
        beta_exp = beta_gate.unsqueeze(-1)  # (batch, seq_len, num_heads, 1)

        a = eigenvalues_exp * alpha_exp  # (batch, seq_len, num_heads, 1)

        # Compute inputs: b_t = beta_t * x_t
        x_heads = x_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b = beta_exp * x_heads  # (batch, seq_len, num_heads, head_dim)

        # Initial hidden state as heads
        h0 = h.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)

        # Cube-first parallel scan
        h_all = self._cube_parallel_scan(a, b, h0)  # (batch, seq_len, num_heads, head_dim)

        # Reshape to full hidden dim
        h_all = h_all.view(batch_size, seq_len, self.hidden_dim)  # (batch, seq_len, hidden_dim)

        # Output projection
        output = self.output_proj(h_all)
        output = self.dropout(output)

        # Final hidden state
        h_final = h_all[:, -1, :]  # (batch, hidden_dim)

        if is_single_step:
            return output.squeeze(1), h_final

        return output, h_final


class DeltaNetLayer(nn.Module):
    """Layer wrapping DeltaNet cell with residual connection."""

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
        """Forward pass with residual connection."""
        residual = self.residual_proj(x)
        x_norm = self.layer_norm(x)
        output, h_new = self.cell(x_norm, h)
        output = output + residual
        return output, h_new


class DeltaNetStack(nn.Module):
    """Stack of DeltaNet layers."""

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
        """Forward pass through all layers."""
        batch_size = x.size(0)
        device = x.device

        if h is None:
            h = torch.zeros(batch_size, self.num_layers, self.layers[0].cell.hidden_dim, device=device, dtype=x.dtype)

        new_h_list = []
        for i, layer in enumerate(self.layers):
            x, h_i = layer(x, h[:, i, :] if h.dim() == 3 else None)
            new_h_list.append(h_i)

        new_h = torch.stack(new_h_list, dim=1)
        return x, new_h


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("DeltaNet Cell (Cube-First Parallel Scan) Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)
    import time

    batch_size = 4
    seq_len = 128
    input_dim = 64
    hidden_dim = 256  # Multiple of 16
    num_heads = 4

    # Test single cell
    print("\n1. Testing DeltaNetCell with Cube-first parallel scan...")
    cell = DeltaNetCell(input_dim, hidden_dim, num_heads, chunk_size=16)

    x = torch.randn(batch_size, seq_len, input_dim)

    start = time.time()
    output, h = cell(x)
    elapsed = time.time() - start

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden state shape: {h.shape}")
    print(f"   Forward time: {elapsed*1000:.2f}ms")
    print(f"   Eigenvalues: {cell.get_eigenvalues()}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Output shape mismatch!"
    assert h.shape == (batch_size, hidden_dim), "Hidden state shape mismatch!"
    print("   [OK] DeltaNetCell test passed!")

    # Verify dimensions are multiples of 16
    print("\n2. Verifying dimension alignment (multiples of 16)...")
    assert hidden_dim % 16 == 0, f"hidden_dim {hidden_dim} not multiple of 16"
    assert (hidden_dim // num_heads) % 16 == 0 or (hidden_dim // num_heads) >= 16, \
        f"head_dim {hidden_dim // num_heads} not optimal for Cube"
    print(f"   hidden_dim: {hidden_dim} (x{hidden_dim // 16})")
    print(f"   head_dim: {hidden_dim // num_heads}")
    print("   [OK] Dimension alignment verified!")

    # Performance test
    print("\n3. Performance test (100 iterations)...")
    start = time.time()
    for _ in range(100):
        output, h = cell(x, h)
    elapsed = time.time() - start
    print(f"   100 iterations: {elapsed:.3f}s ({elapsed*10:.1f}ms per iter)")

    # Test single step
    print("\n4. Testing single-step execution...")
    x_single = torch.randn(batch_size, input_dim)
    output_single, h_new = cell(x_single, h)
    print(f"   Single input shape: {x_single.shape}")
    print(f"   Single output shape: {output_single.shape}")
    assert output_single.shape == (batch_size, hidden_dim), "Single output shape mismatch!"
    print("   [OK] Single-step test passed!")

    # Test layer
    print("\n5. Testing DeltaNetLayer...")
    layer = DeltaNetLayer(input_dim, hidden_dim, num_heads)
    output, h = layer(x)
    print(f"   Layer output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Layer output shape mismatch!"
    print("   [OK] DeltaNetLayer test passed!")

    # Test stack
    print("\n6. Testing DeltaNetStack...")
    stack = DeltaNetStack(input_dim, hidden_dim, num_layers=3, num_heads=num_heads)

    start = time.time()
    output, h_stack = stack(x)
    elapsed = time.time() - start

    print(f"   Stack output shape: {output.shape}")
    print(f"   Stack hidden shape: {h_stack.shape}")
    print(f"   Forward time: {elapsed*1000:.2f}ms")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Stack output shape mismatch!"
    assert h_stack.shape == (batch_size, 3, hidden_dim), "Stack hidden shape mismatch!"
    print("   [OK] DeltaNetStack test passed!")

    # Test eigenvalue constraints
    print("\n7. Testing eigenvalue constraints...")
    eigenvalues = cell.get_eigenvalues()
    assert (eigenvalues >= -1).all() and (eigenvalues <= 1).all(), "Eigenvalues out of bounds!"
    print(f"   Eigenvalues in [-1, 1]: {eigenvalues}")
    print("   [OK] Eigenvalue constraint test passed!")

    # Test gradient flow
    print("\n8. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_count = sum(1 for p in stack.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"   Parameters with gradients: {grad_count}/{len(list(stack.parameters()))}")
    assert grad_count > 0, "No gradients found!"
    print("   [OK] Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in cell.parameters())
    print(f"\n   Total parameters in cell: {total_params:,}")

    print("\n" + "=" * 60)
    print("All DeltaNet (Cube-First) tests passed!")
    print("=" * 60)
