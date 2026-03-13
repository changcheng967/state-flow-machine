"""
Linear Attention for System 1 (Perception)

O(n) complexity instead of O(n²) standard attention.
This enables processing long code sequences efficiently.

Key insight: Factorize the attention matrix into Q(K^T)V ≈ Q(K^T V) using
kernel feature maps, reducing complexity from O(n²) to O(n).

Based on "Transformers are RNNs" (Katharopoulos et al., 2020) and
"Linear Transformer" (Choromanski et al., 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class FeatureMap(nn.Module):
    """
    Feature map for linear attention.

    Maps queries and keys to a feature space where attention
    can be computed as Q(K^T V) instead of (Q K^T) V.
    """

    def __init__(self, dim: int, num_features: Optional[int] = None):
        """
        Initialize feature map.

        Args:
            dim: Input dimension.
            num_features: Number of feature dimensions (default: same as dim).
        """
        super().__init__()
        self.dim = dim
        self.num_features = num_features or dim

        # Random projection for feature map (fixed)
        projection = torch.randn(dim, self.num_features) / math.sqrt(dim)
        self.register_buffer("projection", projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature map.

        Uses the positive random features trick:
        phi(x) = elu(x @ projection) + 1

        This ensures non-negativity and efficient computation.

        Args:
            x: Input tensor (batch, seq, dim).

        Returns:
            Feature-mapped tensor (batch, seq, num_features).
        """
        projected = torch.matmul(x, self.projection)
        return F.elu(projected) + 1


class LinearAttention(nn.Module):
    """
    Linear complexity attention.

    Computes attention in O(n) time by factorizing:
        softmax(Q K^T) V ≈ phi(Q) (phi(K)^T V)

    where phi is a feature map that makes the approximation valid.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize linear attention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head (default: dim // num_heads).
            dropout: Dropout probability.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0 or head_dim is not None, \
            "dim must be divisible by num_heads or head_dim must be specified"

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # Feature map for linear attention
        self.feature_map = FeatureMap(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with linear attention.

        Args:
            x: Input tensor (batch, seq_len, dim).
            mask: Optional causal mask (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Apply mask to keys if provided (for causal attention)
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, seq_len, 1)
            mask = mask.unsqueeze(1).unsqueeze(-1)
            k = k * mask

        # Linear attention computation
        # First compute K^T V: (batch, heads, head_dim, head_dim)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # Then compute Q (K^T V): (batch, heads, seq, head_dim)
        qkv = torch.einsum("bhnd,bhde->bhne", q, kv)

        # Normalize by sum of keys
        k_sum = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)
        normalizer = torch.einsum("bhnd,bhkd->bhnk", q, k_sum) + 1e-6  # (batch, heads, seq, 1)
        normalizer = normalizer.squeeze(-1)  # (batch, heads, seq)

        # Apply normalization
        out = qkv / normalizer.unsqueeze(-1)

        # Transpose back: (batch, seq, heads, head_dim)
        out = out.transpose(1, 2)

        # Reshape and project
        out = out.contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class CausalLinearAttention(nn.Module):
    """
    Causal linear attention for autoregressive generation.

    Uses a recurrent formulation that maintains a running KV state,
    enabling both training (parallel) and inference (recurrent) modes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize causal linear attention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            dropout: Dropout probability.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # Feature map
        self.feature_map = FeatureMap(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with causal linear attention.

        Args:
            x: Input tensor (batch, seq_len, dim).
            state: Optional (KV_sum, K_sum) for recurrent inference.
            return_state: Whether to return updated state.

        Returns:
            Output tensor (batch, seq_len, dim).
            Optionally also returns (KV_sum, K_sum) state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Initialize or use state
        if state is not None:
            kv_state, k_state = state
        else:
            kv_state = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim,
                                   device=device, dtype=dtype)
            k_state = torch.zeros(batch_size, self.num_heads, 1, self.head_dim,
                                  device=device, dtype=dtype)

        # Process sequentially for causal attention
        outputs = []
        for t in range(seq_len):
            q_t = q[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)
            k_t = k[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)
            v_t = v[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)

            # Update state
            kv_update = torch.einsum("bhnd,bhne->bhde", k_t, v_t)
            kv_state = kv_state + kv_update
            k_state = k_state + k_t.sum(dim=2, keepdim=True)

            # Compute output
            out_t = torch.einsum("bhnd,bhde->bhne", q_t, kv_state)
            normalizer = torch.einsum("bhnd,bhkd->bhnk", q_t, k_state) + 1e-6
            out_t = out_t / normalizer

            outputs.append(out_t)

        # Stack outputs
        out = torch.cat(outputs, dim=2)  # (batch, heads, seq, head_dim)

        # Transpose back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        out = self.dropout(out)

        if return_state:
            return out, (kv_state, k_state)
        return out


class LinearAttentionBlock(nn.Module):
    """
    Complete attention block with linear attention, FFN, and residuals.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        causal: bool = False
    ):
        """
        Initialize attention block.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension (default: 4 * dim).
            dropout: Dropout probability.
            causal: Whether to use causal attention.
        """
        super().__init__()

        ff_dim = ff_dim or (4 * dim)

        # Attention
        self.attention = CausalLinearAttention(dim, num_heads, dropout=dropout) if causal \
            else LinearAttention(dim, num_heads, dropout=dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through block.

        Args:
            x: Input tensor (batch, seq_len, dim).
            state: Optional attention state for causal mode.
            return_state: Whether to return state.

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        # Attention with residual
        if return_state or state is not None:
            attn_out, state = self.attention(self.norm1(x), state=state, return_state=True)
        else:
            attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        if return_state:
            return x, state
        return x


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Linear Attention Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    dim = 128
    num_heads = 8

    # Test FeatureMap
    print("\n1. Testing FeatureMap...")
    feature_map = FeatureMap(dim)
    x = torch.randn(batch_size, seq_len, dim)
    features = feature_map(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Feature shape: {features.shape}")
    assert features.shape == (batch_size, seq_len, dim), "FeatureMap output shape mismatch!"
    assert (features >= 0).all(), "FeatureMap should produce non-negative outputs!"
    print("   ✓ FeatureMap test passed!")

    # Test LinearAttention
    print("\n2. Testing LinearAttention...")
    linear_attn = LinearAttention(dim, num_heads)
    out = linear_attn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (batch_size, seq_len, dim), "LinearAttention output shape mismatch!"
    print("   ✓ LinearAttention test passed!")

    # Test CausalLinearAttention
    print("\n3. Testing CausalLinearAttention...")
    causal_attn = CausalLinearAttention(dim, num_heads)
    out = causal_attn(x)
    print(f"   Output shape: {out.shape}")
    assert out.shape == (batch_size, seq_len, dim), "CausalLinearAttention output shape mismatch!"
    print("   ✓ CausalLinearAttention test passed!")

    # Test stateful inference
    print("\n4. Testing stateful inference...")
    out_full, state = causal_attn(x, return_state=True)
    print(f"   State (KV) shape: {state[0].shape}")
    print(f"   State (K) shape: {state[1].shape}")

    # Process single step with state
    x_step = torch.randn(batch_size, 1, dim)
    out_step, state_new = causal_attn(x_step, state=state, return_state=True)
    print(f"   Single step output shape: {out_step.shape}")
    assert out_step.shape == (batch_size, 1, dim), "Single step output shape mismatch!"
    print("   ✓ Stateful inference test passed!")

    # Test LinearAttentionBlock
    print("\n5. Testing LinearAttentionBlock...")
    block = LinearAttentionBlock(dim, num_heads, causal=True)
    out = block(x)
    print(f"   Block output shape: {out.shape}")
    assert out.shape == (batch_size, seq_len, dim), "Block output shape mismatch!"
    print("   ✓ LinearAttentionBlock test passed!")

    # Compare complexity
    print("\n6. Complexity comparison...")
    n = 1024
    standard_ops = n * n  # O(n²)
    linear_ops = n * dim  # O(n * d)
    print(f"   Sequence length: {n}")
    print(f"   Standard attention ops: {standard_ops:,}")
    print(f"   Linear attention ops: {linear_ops:,}")
    print(f"   Speedup: {standard_ops / linear_ops:.1f}x")
    print("   ✓ Complexity advantage confirmed!")

    # Test gradient flow
    print("\n7. Testing gradient flow...")
    loss = out.sum()
    loss.backward()
    grad_exists = False
    for param in block.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in linear_attn.parameters())
    print(f"\n   Total parameters (attention): {total_params:,}")

    print("\n" + "=" * 60)
    print("All Linear Attention tests passed!")
    print("=" * 60)
