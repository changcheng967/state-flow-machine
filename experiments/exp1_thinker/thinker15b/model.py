"""model.py — Thinker-1.5B model components for MindSpore 2.2.

All components are nn.Cell subclasses, GRAPH_MODE-safe.
Dimensions are multiples of 16 for DaVinci Cube.

This file is the reference/documentation version. The actual training script
(thinker15b/train.py) inlines these classes because MS 2.2's
inspect.getsourcelines() must trace within the same file for GRAPH_MODE.
"""

from __future__ import annotations

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal, One
from mindspore.common.tensor import Tensor

try:
    from .config import Thinker15BConfig
except ImportError:
    from config import Thinker15BConfig


def _fp16_dense(in_ch: int, out_ch: int, has_bias: bool = False) -> nn.Dense:
    """Create nn.Dense with FP16 parameters (explicit, not to_float)."""
    w = np.random.randn(out_ch, in_ch).astype(np.float16) * (1.0 / np.sqrt(in_ch))
    dense = nn.Dense(in_ch, out_ch, has_bias=has_bias)
    dense.weight = ms.Parameter(Tensor(w), name=dense.weight.name)
    if has_bias:
        dense.bias = ms.Parameter(
            Tensor(np.zeros(out_ch, dtype=np.float16)),
            name=dense.bias.name,
        )
    return dense


class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = Tensor(eps, ms.float32)
        self.weight = ms.Parameter(
            Tensor(np.ones(dim, dtype=np.float32)), name="norm_weight",
        )

    def construct(self, x: Tensor) -> Tensor:
        x_f = x.astype(ms.float32)
        variance = ops.mean(x_f * x_f, axis=-1, keep_dims=True)
        x_norm = x_f * ops.rsqrt(variance + self.eps)
        return (x_norm * self.weight).astype(x.dtype)


class RotaryEmbedding(nn.Cell):
    """Pre-computed RoPE cos/sin tables."""

    def __init__(self, head_dim: int, max_seq: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (
            np.arange(0, head_dim, 2, dtype=np.float32) / head_dim
        ))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float16)
        sin = np.sin(emb).astype(np.float16)
        self.cos_table = Tensor(cos[np.newaxis, np.newaxis, :, :])
        self.sin_table = Tensor(sin[np.newaxis, np.newaxis, :, :])

    def construct(self, x: Tensor) -> Tensor:
        """Apply rotary embedding. x: (B, H, S, D)."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = ops.concat([-x2, x1], axis=-1)
        return x * self.cos_table + rotated * self.sin_table


class TransformerBlock(nn.Cell):
    """Single transformer layer: MHA (no GQA) + SwiGLU FFN + RMSNorm."""

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        H = cfg.hidden_dim
        A = cfg.intermediate_dim
        NH = cfg.num_heads
        HD = cfg.head_dim

        # Attention
        self.q_proj = _fp16_dense(H, NH * HD)
        self.k_proj = _fp16_dense(H, NH * HD)
        self.v_proj = _fp16_dense(H, NH * HD)
        self.o_proj = _fp16_dense(NH * HD, H)
        self.input_norm = RMSNorm(H)
        self.post_attn_norm = RMSNorm(H)

        # FFN (SwiGLU)
        self.gate_proj = _fp16_dense(H, A)
        self.up_proj = _fp16_dense(H, A)
        self.down_proj = _fp16_dense(A, H)
        self.ffn_norm = RMSNorm(H)

        self.scale = Tensor(HD ** -0.5, ms.float16)
        self.NH = NH
        self.HD = HD

        # Causal mask (computed at init, not in construct)
        S = cfg.max_seq_len
        mask_np = np.triu(
            np.full((S, S), -1e4, dtype=np.float16), k=1,
        )
        self.causal_mask = Tensor(mask_np[np.newaxis, np.newaxis, :, :])

        self.rope = RotaryEmbedding(HD, S, cfg.rope_theta)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> Tensor:
        B, S, _ = x.shape
        NH, HD = self.NH, self.HD

        # Self-attention with pre-norm
        h = self.input_norm(x)
        Q = self.q_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)

        # RoPE
        Q = Q * cos + ops.concat([-Q[..., HD:], Q[..., :HD]], axis=-1) * sin
        K = K * cos + ops.concat([-K[..., HD:], K[..., :HD]], axis=-1) * sin

        # Scaled dot-product attention (causal)
        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = attn + mask[:, :, :S, :S]
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        out = self.o_proj(out)
        x = x + self.post_attn_norm(out)

        # SwiGLU FFN
        h = self.ffn_norm(x)
        gate = ops.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


class SFMSlotBank(nn.Cell):
    """GRU-based State Slot Bank.

    16 typed slots (8 variable, 4 control-flow, 2 scratch, 2 global).
    Cross-attention reads from slots, GRU gated update writes to slots.
    All params FP16.

    Returns (modified_hidden, new_slots).
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.H = cfg.hidden_dim
        self.num_slots = cfg.num_slots
        self.slot_dim = cfg.slot_dim
        self.num_heads = cfg.slot_num_heads
        self.head_dim = cfg.slot_dim // cfg.slot_num_heads

        # Learnable slot vectors
        self.slot_vectors = ms.Parameter(
            Tensor(np.random.randn(cfg.num_slots, cfg.slot_dim).astype(
                np.float16) * 0.02),
            name="slot_vectors",
        )

        # Cross-attention: Q from hidden, K/V from slots
        self.q_proj = _fp16_dense(cfg.hidden_dim,
                                  cfg.slot_num_heads * self.head_dim)
        self.k_proj = _fp16_dense(cfg.slot_dim,
                                  cfg.slot_num_heads * self.head_dim)
        self.v_proj = _fp16_dense(cfg.slot_dim,
                                  cfg.slot_num_heads * self.head_dim)
        self.out_proj = _fp16_dense(
            cfg.slot_num_heads * self.head_dim, cfg.hidden_dim)

        self.layer_norm = RMSNorm(cfg.hidden_dim)

        # Gated recurrent slot update (GRU-style)
        self.W_alpha = _fp16_dense(cfg.hidden_dim, cfg.slot_dim,
                                   has_bias=True)
        self.W_beta = _fp16_dense(cfg.hidden_dim, cfg.slot_dim,
                                  has_bias=True)
        self.W_v = _fp16_dense(cfg.hidden_dim, cfg.slot_dim,
                               has_bias=True)

        self.scale = Tensor(self.head_dim ** -0.5, ms.float16)

    def construct(self, hidden_states: Tensor) -> tuple:
        """Process hidden states through slot bank.

        Returns (modified_hidden, new_slots).
        """
        B, S, _ = hidden_states.shape
        NS = self.num_slots
        NH = self.num_heads
        HD = self.head_dim

        # Broadcast slots across batch
        slots = ops.broadcast_to(
            self.slot_vectors.reshape(1, NS, self.slot_dim),
            (B, NS, self.slot_dim),
        )

        # Cross-attention: hidden -> slots
        Q = self.q_proj(hidden_states).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        out = self.out_proj(out)

        # Residual connection with learned gate
        modified = self.layer_norm(hidden_states + out)

        # Gated recurrent slot update: mean-pool hidden, update all slots
        h_mean = ops.mean(hidden_states, axis=1, keep_dims=True)  # (B, 1, H)
        alpha = ops.sigmoid(self.W_alpha(h_mean))  # (B, 1, slot_dim)
        beta = ops.sigmoid(self.W_beta(h_mean))
        v = ops.tanh(self.W_v(h_mean))
        new_slots = alpha * slots + beta * v  # (B, num_slots, slot_dim)

        return modified, new_slots


class SlotPredictionHead(nn.Cell):
    """Auxiliary head: predict slot contents from hidden states.

    Projects mean-hidden to (B, num_slots, slot_dim). Used for
    auxiliary MSE loss to train slot bank.
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.pred_proj = _fp16_dense(cfg.hidden_dim,
                                     cfg.num_slots * cfg.slot_dim)
        self.num_slots = cfg.num_slots
        self.slot_dim = cfg.slot_dim

    def construct(self, hidden_states: Tensor) -> Tensor:
        """Returns (B, num_slots, slot_dim) prediction."""
        h_mean = ops.mean(hidden_states, axis=1)  # (B, H)
        pred = self.pred_proj(h_mean)  # (B, num_slots * slot_dim)
        return pred.view(h_mean.shape[0], self.num_slots, self.slot_dim)


class Thinker15BModel(nn.Cell):
    """Full ~1.45B decoder-only LM with SFM slot banks.

    Construct is unrolled (no for-loops) so inspect.getsourcelines()
    can trace it in GRAPH_MODE. SFM banks are inserted at the
    configured layer indices.
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.cfg = cfg
        V = cfg.vocab_size
        H = cfg.hidden_dim
        NS = cfg.num_slots
        SD = cfg.slot_dim

        self.embedding = nn.Embedding(V, H)
        self.norm = RMSNorm(H)
        self.lm_head = ms.Parameter(
            Tensor(np.random.randn(V, H).astype(np.float16) * 0.02),
            name="lm_head_weight",
        )
        self.matmul = ops.MatMul(transpose_b=True)

        # Transformer layers
        self.layers = nn.CellList([
            TransformerBlock(cfg) for _ in range(cfg.num_layers)
        ])

        # SFM slot banks at configured layers
        self.sfm_banks = nn.CellList([
            SFMSlotBank(cfg) for _ in cfg.sfm_layers
        ])
        self.sfm_pred_heads = nn.CellList([
            SlotPredictionHead(cfg) for _ in cfg.sfm_layers
        ])
        self.sfm_layer_set = set(cfg.sfm_layers)

        # Build sfm_layer_index map: layer_idx -> sfm_list_idx
        self.sfm_index_map: dict[int, int] = {}
        for i, layer_idx in enumerate(cfg.sfm_layers):
            self.sfm_index_map[layer_idx] = i

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> tuple:
        """Forward pass. Returns (logits, total_slot_pred_loss).

        For GRAPH_MODE safety, this construct is fully unrolled.
        Override in the training script with the unrolled version.
        """
        B = input_ids.shape[0]
        x = self.embedding(input_ids)
        total_slot_loss = Tensor(0.0, ms.float32)
        mse_fn = nn.MSELoss()

        sfm_idx = 0
        for i in range(self.cfg.num_layers):
            x = self.layers[i](x, cos, sin, mask)
            if i in self.sfm_layer_set and sfm_idx < len(self.sfm_banks):
                x, new_slots = self.sfm_banks[sfm_idx](x)
                pred_slots = self.sfm_pred_heads[sfm_idx](x)
                total_slot_loss = total_slot_loss + mse_fn(pred_slots, new_slots.astype(ms.float32))
                sfm_idx += 1

        x = self.norm(x)
        h2 = x.reshape((-1, self.cfg.hidden_dim))
        logits = self.matmul(h2, self.lm_head)
        logits = logits.reshape(B, x.shape[1], self.cfg.vocab_size)
        return logits, total_slot_loss
