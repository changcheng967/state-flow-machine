"""
DeltaNet Recurrent Cell - GATED DELTANET with CUBE-FIRST PARALLEL SCAN

The core component enabling state tracking. Uses eigenvalues in [-1, 1]
(not [0, 1] like standard RNNs) which enables tracking state transformations
that transformers cannot do (TC0 circuit complexity limit).

Key insight: Negative eigenvalues allow the cell to "subtract" state,
enabling reversible computations and proper tracking of variable mutations.

GATED DELTANET (ICLR 2025):
- Forget gate controls how much old state to retain vs new input to inject
- forget≈1: retains old state (normal tracking)
- forget≈0: replaces state with new value (variable reassignment)
- Boosts accuracy by enabling proper state overwrite on reassignment

CUBE-FIRST OPTIMIZATION for Ascend NPU:
- Reshape sequence into chunks of size 16 (matches Cube unit width)
- Within-chunk recurrence uses MATRIX MULTIPLICATION (runs on Cube)
- Between-chunk carry uses Vector unit
- This gives Cube+Vector pipeline parallelism

OPTIMIZATIONS (2024-03):
- Selective FP32 for cumprod operations (Vector unit anyway, no Cube cost)
- Log-space computation for numerical stability
- Batched chunk processing (single Cube matmul for all chunks)
- Avoid torch.where to prevent AICPU fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DeltaNetCell(nn.Module):
    """
    Gated DeltaNet recurrent cell with learnable eigenvalues in [-1, 1].

    CUBE-FIRST: Uses matrix-based parallel scan within 16-step chunks.
    The 16x16 lower-triangular matrix M is computed and multiplied,
    utilizing the DaVinci Cube unit efficiently.

    GATED: Forget gate (per-head) controls state retention vs injection,
    enabling proper variable reassignment tracking.

    OPTIMIZED: Uses selective FP32 for numerically sensitive operations
    (cumprod, division) which run on Vector unit anyway.
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

        # Forget gate: controls state retention (1=keep old, 0=replace)
        self.forget_gate_proj = nn.Linear(input_dim, num_heads, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Eigenvalue parameters (one per head) - constrained to [-1, 1] via tanh
        self.eigenvalue_raw = nn.Parameter(torch.zeros(num_heads))
        self._init_eigenvalues(eigenvalue_init)

        # Layer norms
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _init_eigenvalues(self, init_value: float):
        """Initialize eigenvalue parameters."""
        if abs(init_value) >= 1:
            init_value = 0.99 * (1 if init_value > 0 else -1)
        raw_value = 0.5 * math.log((1 + init_value) / (1 - init_value))
        nn.init.constant_(self.eigenvalue_raw, raw_value)

    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues constrained to [-1, 1]."""
        return torch.tanh(self.eigenvalue_raw)

    def _compute_chunk_matrix_batched(self, a_chunks: torch.Tensor) -> torch.Tensor:
        """
        Compute the 16x16 lower-triangular matrix M for ALL chunks at once.

        Uses LOG-SPACE computation for numerical stability in FP16/AMP:
        - log(cumprod) = cumsum(log(a)) — stable, no overflow
        - M[i,j] = exp(log_cumprod[i] - log_cumprod[j-1])

        This avoids the cumprod → divide pattern that overflows FP16.

        Args:
            a_chunks: (batch, num_chunks, chunk_len, num_heads) - decay factors

        Returns:
            M_all: (batch*num_chunks, chunk_len, chunk_len, num_heads) - lower triangular matrices
        """
        batch_size, num_chunks, chunk_len, num_heads = a_chunks.shape
        device = a_chunks.device
        input_dtype = a_chunks.dtype

        # Flatten batch and chunks for unified processing
        a_flat = a_chunks.reshape(batch_size * num_chunks, chunk_len, num_heads)

        # === SELECTIVE FP32: Log-space computation for stability ===
        # These ops run on Vector unit anyway, so FP32 costs zero Cube throughput
        a_fp32 = a_flat.float()

        # Handle negative eigenvalues: track sign separately
        # Use mask multiplication instead of torch.where (prevents AICPU fallback)
        sign_a = a_fp32.sign()
        zero_mask = (sign_a == 0).float()
        sign_a = sign_a + zero_mask  # Replace zeros with 1.0

        # Log-space cumprod: log(cumprod) = cumsum(log(|a|))
        log_a = torch.log(a_fp32.abs().clamp(min=1e-7))  # Avoid log(0)
        log_cumprod = torch.cumsum(log_a, dim=1)  # (B*C, chunk_len, num_heads)

        # For M[i,j] = cumprod[i] / cumprod[j-1], in log space:
        # log(M[i,j]) = log_cumprod[i] - log_cumprod[j-1]
        # Pad log_cumprod with 0 at position -1 (log(1) = 0)
        log_cumprod_padded = F.pad(log_cumprod, (0, 0, 1, 0), value=0.0)

        # Broadcasting: (B*C, chunk_len, 1, nh) - (B*C, 1, chunk_len, nh)
        log_cumprod_i = log_cumprod.unsqueeze(2)  # (B*C, chunk_len, 1, num_heads)
        log_cumprod_j = log_cumprod_padded[:, :-1, :].unsqueeze(1)  # (B*C, 1, chunk_len, num_heads)

        log_M = log_cumprod_i - log_cumprod_j  # (B*C, chunk_len, chunk_len, num_heads)
        M = torch.exp(log_M)  # Back to linear space

        # Handle signs: sign(cumprod) = cumprod(sign)
        sign_cumprod = torch.cumprod(sign_a, dim=1)
        sign_cumprod_padded = F.pad(sign_cumprod, (0, 0, 1, 0), value=1.0)
        sign_M = sign_cumprod.unsqueeze(2) * sign_cumprod_padded[:, :-1, :].unsqueeze(1)
        M = M * sign_M

        # === AVOID torch.where: Use mask multiplication (prevents AICPU fallback) ===
        # Lower triangular mask
        tril_mask = torch.tril(torch.ones(chunk_len, chunk_len, device=device, dtype=torch.float32))
        tril_mask = tril_mask.unsqueeze(0).unsqueeze(-1)  # (1, chunk_len, chunk_len, 1)
        M = M * tril_mask

        # Diagonal mask: set diagonal to 1
        diag_mask = torch.eye(chunk_len, device=device, dtype=torch.float32)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1)  # (1, chunk_len, chunk_len, 1)
        M = M * (1.0 - diag_mask) + diag_mask

        # Cast back to input dtype for Cube matmul
        return M.to(input_dtype)  # (B*C, chunk_len, chunk_len, num_heads)

    def _cube_parallel_scan(
        self,
        a: torch.Tensor,  # (batch, seq_len, num_heads, 1) - decay factors
        b: torch.Tensor,  # (batch, seq_len, num_heads, head_dim) - inputs
        h0: torch.Tensor  # (batch, num_heads, head_dim) - initial state
    ) -> torch.Tensor:
        """
        Cube-first parallel scan for linear recurrence: h_t = a_t * h_{t-1} + b_t

        OPTIMIZED: Batched chunk processing — ONE Cube matmul for ALL chunks,
        then sequential carry for inter-chunk dependencies.

        Uses log-space computation for numerical stability under AMP/FP16.
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

        # === BATCHED: Compute M for ALL chunks at once (one kernel launch) ===
        M_all = self._compute_chunk_matrix_batched(a_chunks)  # (B*C, cs, cs, nh)

        # === BATCHED CUBE: Single matmul for all chunks ===
        # M_all: (B*C, cs, cs, nh) -> (B*C*nh, cs, cs)
        M_flat = M_all.permute(0, 3, 1, 2).reshape(
            batch_size * num_chunks * num_heads, chunk_size, chunk_size
        )
        # B_chunks: (B, C, cs, nh, hd) -> (B*C*nh, hd, cs).T -> (B*C*nh, cs, hd)
        B_flat = b_chunks.permute(0, 1, 3, 4, 2).reshape(
            batch_size * num_chunks * num_heads, head_dim, chunk_size
        ).transpose(1, 2)

        # ONE Cube matmul for ALL chunks
        h_contrib_all = torch.bmm(M_flat, B_flat)  # (B*C*nh, cs, hd)
        h_contrib_all = h_contrib_all.view(
            batch_size, num_chunks, num_heads, chunk_size, head_dim
        ).permute(0, 1, 3, 2, 4)  # (batch, num_chunks, chunk_size, num_heads, head_dim)

        # === SELECTIVE FP32: Cumprod for h0 contribution ===
        a_fp32 = a_chunks.float()
        cum_a_all = torch.cumprod(a_fp32, dim=2)  # (batch, num_chunks, chunk_size, num_heads)

        # === SEQUENTIAL: Inter-chunk carry (only num_chunks iterations, not seq_len) ===
        h = h0
        outputs = []

        for c in range(num_chunks):
            # h0 contribution: cum_a * h_prev (cast back to input dtype for consistency)
            h0_contrib = cum_a_all[:, c, :, :].unsqueeze(-1).to(dtype) * h.unsqueeze(1)
            # Total: h_chunk = h0_contrib + intra_chunk_contribution
            h_chunk = h0_contrib + h_contrib_all[:, c]  # (batch, chunk_size, num_heads, head_dim)
            outputs.append(h_chunk)
            # Carry: last hidden state for next chunk
            h = h_chunk[:, -1, :, :]

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

        # GATED: Forget gate controls state retention vs injection
        # forget≈1: retain old state (normal tracking)
        # forget≈0: replace with new value (variable reassignment)
        forget = torch.sigmoid(self.forget_gate_proj(x))  # (batch, seq_len, num_heads)

        # Compute decay factors: a_t = forget * eigenvalue * alpha_t
        eigenvalues_exp = eigenvalues.view(1, 1, self.num_heads, 1)  # (1, 1, num_heads, 1)
        alpha_exp = alpha_gate.unsqueeze(-1)  # (batch, seq_len, num_heads, 1)
        beta_exp = beta_gate.unsqueeze(-1)  # (batch, seq_len, num_heads, 1)
        forget_exp = forget.unsqueeze(-1)  # (batch, seq_len, num_heads, 1)

        a = forget_exp * eigenvalues_exp * alpha_exp  # (batch, seq_len, num_heads, 1)

        # Compute inputs: b_t = (1 - forget) * beta_t * x_t
        x_heads = x_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b = (1 - forget_exp) * beta_exp * x_heads  # (batch, seq_len, num_heads, head_dim)

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
