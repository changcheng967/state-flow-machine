"""model.py — Thinker-1.5B v2 model components for MindSpore 2.2.

All components are nn.Cell subclasses, GRAPH_MODE-safe.
Dimensions are multiples of 16 for DaVinci Cube.

v2 changes:
  - Real GRU with reset gate for SFM slot bank
  - Slot persistence across all 4 SFM layers (threaded through construct)
  - Standard pre-norm TransformerBlock (removed post_attn_norm)
  - Slot prediction head uses CE loss over 512-token slot vocabulary
  - Initial slot vectors as learned parameter

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
    """Single transformer layer: MHA + SwiGLU FFN + RMSNorm (pre-norm only).

    v2: standard pre-norm — no post_attn_norm. Just x = x + attn_out.
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        H = cfg.hidden_dim
        A = cfg.intermediate_dim
        NH = cfg.num_heads
        HD = cfg.head_dim

        self.q_proj = _fp16_dense(H, NH * HD)
        self.k_proj = _fp16_dense(H, NH * HD)
        self.v_proj = _fp16_dense(H, NH * HD)
        self.o_proj = _fp16_dense(NH * HD, H)
        self.input_norm = RMSNorm(H)

        self.gate_proj = _fp16_dense(H, A)
        self.up_proj = _fp16_dense(H, A)
        self.down_proj = _fp16_dense(A, H)
        self.ffn_norm = RMSNorm(H)

        self.scale = Tensor(HD ** -0.5, ms.float16)
        self.NH = NH
        self.HD = HD

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

        h = self.input_norm(x)
        Q = self.q_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)

        HD2 = self.HD // 2
        Q = Q * cos + ops.concat([-Q[..., HD2:], Q[..., :HD2]], axis=-1) * sin
        K = K * cos + ops.concat([-K[..., HD2:], K[..., :HD2]], axis=-1) * sin

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = attn + mask[:, :, :S, :S]
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        out = self.o_proj(out)
        x = x + out  # v2: no post_attn_norm

        h = self.ffn_norm(x)
        gate = ops.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


class SFMSlotBank(nn.Cell):
    """GRU-based State Slot Bank with real GRU update.

    v2 changes:
    - Real GRU with reset gate:
        z = sigmoid(W_z * [slots; read_content])
        r = sigmoid(W_r * [slots; read_content])
        h_cand = tanh(W_h * [r * slots; read_content])
        new_slots = (1-z) * slots + z * h_cand
    - Cross-attention: Q from hidden, K/V from slots (same as v1)
    - Returns (modified_hidden, new_slots)
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.H = cfg.hidden_dim
        self.num_slots = cfg.num_slots
        self.slot_dim = cfg.slot_dim
        self.num_heads = cfg.slot_num_heads
        self.head_dim = cfg.slot_dim // cfg.slot_num_heads

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

        # Real GRU: 3 gates (update, reset, candidate) — all have bias
        # Input: concat(slots, mean_hidden) -> slot_dim
        gru_input_dim = cfg.slot_dim + cfg.hidden_dim
        self.W_z = _fp16_dense(gru_input_dim, cfg.slot_dim, has_bias=True)
        self.W_r = _fp16_dense(gru_input_dim, cfg.slot_dim, has_bias=True)
        self.W_h = _fp16_dense(gru_input_dim, cfg.slot_dim, has_bias=True)

        self.scale = Tensor(self.head_dim ** -0.5, ms.float16)

    def construct(self, hidden_states: Tensor, slots: Tensor) -> tuple:
        """Process hidden states through slot bank with GRU update.

        Args:
            hidden_states: (B, S, H) from previous transformer layer.
            slots: (B, num_slots, slot_dim) — current slot state.

        Returns:
            (modified_hidden, new_slots) — both (B, S, H) and (B, NS, SD).
        """
        B, S, _ = hidden_states.shape
        NS = self.num_slots
        NH = self.num_heads
        HD = self.head_dim

        # Cross-attention: hidden -> slots
        Q = self.q_proj(hidden_states).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        read_content = self.out_proj(out)

        # Residual connection
        modified = self.layer_norm(hidden_states + read_content)

        # GRU slot update: mean-pool hidden, update all slots
        h_mean = ops.mean(hidden_states, axis=1, keep_dims=True)  # (B, 1, H)
        h_mean_broad = ops.broadcast_to(h_mean, (B, NS, self.H))

        # Concatenate [slots; h_mean] -> (B, NS, slot_dim + H)
        gru_input = ops.concat([slots, h_mean_broad], axis=-1)

        z = ops.sigmoid(self.W_z(gru_input))    # update gate
        r = ops.sigmoid(self.W_r(gru_input))    # reset gate
        h_cand = ops.tanh(self.W_h(ops.concat([r * slots, h_mean_broad], axis=-1)))
        new_slots = (1.0 - z) * slots + z * h_cand

        return modified, new_slots


class SlotPredictionHead(nn.Cell):
    """Predict slot contents discretized to slot vocabulary.

    v2: Projects slots to (B, num_slots, slot_vocab_size) for CE loss.
    Each slot_dim vector is quantized to nearest slot vocab entry.
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.num_slots = cfg.num_slots
        self.slot_dim = cfg.slot_dim
        self.slot_vocab_size = cfg.slot_vocab_size

        # Linear projection: slot_dim -> slot_vocab_size
        self.proj = _fp16_dense(cfg.slot_dim, cfg.slot_vocab_size)

    def construct(self, slots: Tensor) -> Tensor:
        """Returns (B, num_slots, slot_vocab_size) logits."""
        # slots: (B, num_slots, slot_dim)
        logits = self.proj(slots)
        return logits


class SlotTokenizer(nn.Cell):
    """Discretize continuous slot vectors to slot vocabulary IDs.

    Maps (B, num_slots, slot_dim) -> (B, num_slots) integer IDs
    by finding nearest entry in a learned slot vocabulary embedding.
    """

    def __init__(self, cfg: Thinker15BConfig):
        super().__init__()
        self.slot_vocab_size = cfg.slot_vocab_size
        self.slot_dim = cfg.slot_dim
        # Learnable slot vocabulary: (slot_vocab_size, slot_dim)
        self.vocab_vectors = ms.Parameter(
            Tensor(np.random.randn(cfg.slot_vocab_size, cfg.slot_dim).astype(
                np.float16) * 0.02),
            name="slot_vocab_vectors",
        )

    def construct(self, slots: Tensor) -> Tensor:
        """Returns (B, num_slots) integer token IDs."""
        # slots: (B, NS, SD), vocab: (SV, SD)
        # Compute cosine similarity via matmul
        slots_norm = slots / (ops.norm(slots, axis=-1, keep_dims=True) + 1e-8)
        vocab_norm = self.vocab_vectors / (
            ops.norm(self.vocab_vectors, axis=-1, keep_dims=True) + 1e-8)
        # (B, NS, SD) @ (SD, SV) -> (B, NS, SV)
        sim = ops.matmul(slots_norm, vocab_norm.transpose(1, 0))
        return ops.argmax(sim, axis=-1)  # (B, NS)


class Thinker15BModel(nn.Cell):
    """Full ~1.45B decoder-only LM with SFM slot banks.

    v2: Slot persistence across all 4 SFM layers. Slots thread through
    the unrolled construct. Initial slots are a learned parameter.

    Construct is unrolled (no for-loops) for GRAPH_MODE safety.
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

        # Initial slots as learned parameter
        self.initial_slots = ms.Parameter(
            Tensor(np.random.randn(1, NS, SD).astype(np.float16) * 0.02),
            name="initial_slots",
        )

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
        self.sfm_tokenizers = nn.CellList([
            SlotTokenizer(cfg) for _ in cfg.sfm_layers
        ])
        self.sfm_layer_set = set(cfg.sfm_layers)

        self.sfm_index_map: dict[int, int] = {}
        for i, layer_idx in enumerate(cfg.sfm_layers):
            self.sfm_index_map[layer_idx] = i

        self.ce_loss = nn.CrossEntropyLoss()

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> tuple:
        """Forward pass. Returns (logits, total_slot_pred_loss).

        For GRAPH_MODE safety, this construct is fully unrolled.
        """
        B = input_ids.shape[0]
        x = self.embedding(input_ids).astype(ms.float16)
        total_slot_loss = Tensor(0.0, ms.float32)

        # Initialize slots from learned parameter, broadcast to batch
        slots = ops.broadcast_to(
            self.initial_slots, (B, self.cfg.num_slots, self.cfg.slot_dim))

        sfm_idx = 0
        for i in range(self.cfg.num_layers):
            x = self.layers[i](x, cos, sin, mask)
            if i in self.sfm_layer_set and sfm_idx < len(self.sfm_banks):
                x, slots = self.sfm_banks[sfm_idx](x, slots)
                # Slot prediction: CE loss over slot vocabulary
                pred_logits = self.sfm_pred_heads[sfm_idx](slots)
                slot_ids = self.sfm_tokenizers[sfm_idx](slots)
                # Flatten for CE: (B*NS, SV) and (B*NS,)
                B_actual = slots.shape[0]
                flat_logits = pred_logits.reshape((-1, self.cfg.slot_vocab_size))
                flat_ids = slot_ids.reshape((-1,))
                loss = self.ce_loss(flat_logits, flat_ids)
                total_slot_loss = total_slot_loss + loss
                sfm_idx += 1

        x = self.norm(x)
        h2 = x.reshape((-1, self.cfg.hidden_dim))
        logits = self.matmul(h2, self.lm_head)
        logits = logits.reshape(B, x.shape[1], self.cfg.vocab_size)
        return logits, total_slot_loss
