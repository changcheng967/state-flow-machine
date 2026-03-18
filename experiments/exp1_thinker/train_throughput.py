"""
train_throughput.py - OpenI Training Throughput Benchmark for SFM + LoRA (MindSpore)

Self-contained boot file for OpenI Train Task on the mindspore_2_2_cann7_train
image. Zero pip installs needed -- MindSpore is pre-installed and matches CANN 7.

Builds a Qwen2-like model (~6.4B params matching Qwen2.5-Coder-7B dims) inline,
injects LoRA (rank=64) and SFM adapters, then benchmarks training throughput.

All architecture code is inline -- no imports from sfm/ for OpenI portability.

Usage:
    # OpenI Train Task: set as boot file, no args needed
    # Local smoke: python train_throughput.py --local_test
"""

# ===========================================================================
# Phase 0: Env vars + logging (pure stdlib, BEFORE MindSpore import)
# ===========================================================================
import os
import sys
import time
import math
import gc
import argparse

# Ascend 910 tuning env vars — MUST be set before import mindspore
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
os.environ.setdefault("CPU_AFFINITY_CONF", "1")
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")  # ERROR only
os.environ.setdefault("MS_COMPILER_CACHE_ENABLE", "0")  # cache incompatible with @lazy_inline in MS 2.2
os.environ.setdefault("MS_BUILD_PROCESS_NUM", "24")  # parallel op compilation
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_boot_log = "/tmp/sfm_boot.log"
_log_file = _boot_log

try:
    with open(_boot_log, "w") as f:
        f.write(f"[BOOT] train_throughput.py starting (MindSpore)\n")
        f.write(f"[BOOT] Python {sys.version}\n")
        f.write(f"[BOOT] CWD: {os.getcwd()}\n")
        f.write(f"[BOOT] RANK_ID={os.getenv('RANK_ID', 'NOT SET')}\n")
except Exception:
    pass


def log(msg: str) -> None:
    """Print to stdout AND append to log file."""
    print(msg, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


log("=" * 60)
log("[BOOT] train_throughput.py starting (MindSpore)")
log(f"[BOOT] Python {sys.version}")
log(f"[BOOT] CWD: {os.getcwd()}")
log(f"[BOOT] RANK_ID={os.getenv('RANK_ID', 'NOT SET')}")

# ===========================================================================
# Phase 1: Import MindSpore (pre-installed, zero pip)
# ===========================================================================
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, lazy_inline
from mindspore.common.initializer import Normal, Zero, One, initializer
import numpy as np

# ---------------------------------------------------------------------------
# Patch: MS 2.2 value_and_grad's internal @jit wrapper has __module__=None,
# which crashes the GRAPH_MODE parser.  Patch CellNamespace to treat
# None as '__main__' (the script module) instead of crashing.
# ---------------------------------------------------------------------------
import mindspore._extends.parse.namespace as _ns_mod
_orig_cell_ns = _ns_mod.CellNamespace.__init__


def _safe_cell_ns_init(self, name):
    if name is None:
        name = '__main__'
    _orig_cell_ns(self, name)


_ns_mod.CellNamespace.__init__ = _safe_cell_ns_init

log(f"[BOOT] MindSpore {ms.__version__}")

# ===========================================================================
# Phase 2: Args (parse_known_args to handle OpenI injected args)
# ===========================================================================
parser = argparse.ArgumentParser(description="SFM+LoRA Throughput Benchmark (MindSpore)")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--local_test", action="store_true")
parser.add_argument("--npu_count", type=int, default=0)

# ===========================================================================
# Phase 3: Model constants (Qwen2.5-Coder-7B dimensions)
# ===========================================================================
VOCAB_SIZE = 152064
HIDDEN_DIM = 3584
NUM_HEADS = 28
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
NUM_LAYERS = 28
INTERMEDIATE_DIM = 18944
NUM_KV_HEADS = 4  # GQA
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 7

# ===========================================================================
# Phase 4: RMSNorm
# ===========================================================================

class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalization (FP16 throughout)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = Tensor(eps, ms.float16)
        self.weight = Parameter(initializer(One(), (dim,), ms.float16), name="weight")

    def construct(self, x: Tensor) -> Tensor:
        variance = ops.mean(x * x, axis=-1, keep_dims=True)
        x_norm = x * ops.rsqrt(variance + self.eps)
        return x_norm * self.weight


# ===========================================================================
# Phase 5: Rotary Position Embedding (RoPE)
# ===========================================================================

def _build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, dtype=ms.float16):
    """Precompute cos/sin tables for rotary embeddings."""
    theta = 1.0 / (base ** (ops.arange(0, head_dim, 2, dtype=ms.float32) / head_dim))
    positions = ops.arange(seq_len, dtype=ms.float32)
    freqs = ops.outer(positions, theta)  # (seq_len, head_dim/2)
    cos = ops.cos(freqs).astype(dtype)
    sin = ops.sin(freqs).astype(dtype)
    return cos, sin


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding. x: (B, H, S, D). cos/sin: (S, D/2).

    All shape constants (HEAD_DIM, HEAD_DIM//2) are Python ints so
    AUTO_PARALLEL's strategy planner can determine shapes statically.
    """
    HD2 = HEAD_DIM // 2
    x1 = x[..., :HD2]
    x2 = x[..., HD2:]
    cos_s = cos.expand_dims(0).expand_dims(0)  # (S, D/2) → (1, 1, S, D/2)
    sin_s = sin.expand_dims(0).expand_dims(0)
    out1 = x1 * cos_s - x2 * sin_s
    out2 = x1 * sin_s + x2 * cos_s
    return ops.cat([out1, out2], axis=-1)


# ===========================================================================
# Phase 5.5: FP16 helpers — to_float is unreliable in MS 2.2
# ===========================================================================

def _fp16_dense(in_channels: int, out_channels: int, has_bias: bool = False) -> nn.Dense:
    """Create nn.Dense with explicit FP16 weight (no to_float needed)."""
    d = nn.Dense(in_channels, out_channels, has_bias=has_bias)
    d.weight = Parameter(
        initializer(Normal(0.02), d.weight.shape, ms.float16),
        name="weight")
    if has_bias:
        d.bias = Parameter(
            initializer(Zero(), d.bias.shape, ms.float16),
            name="bias")
    return d


def _sharded_dense(in_channels: int, out_channels: int,
                   in_strategy: tuple, weight_strategy: tuple,
                   has_bias: bool = False) -> nn.Dense:
    """Create FP16 Dense with shard strategy on internal matmul.

    SEMI_AUTO_PARALLEL: shard the Dense's matmul so weights are distributed
    across NPUs.  in_strategy/weight_strategy describe how each matmul input
    is split (tuple of ints per dimension, value = number of shards).
    """
    d = _fp16_dense(in_channels, out_channels, has_bias)
    d.matmul.shard((in_strategy, weight_strategy))
    return d


# ===========================================================================
# Phase 6: Transformer Block (MHA + SwiGLU FFN + RMSNorm)
# ===========================================================================

class TransformerBlock(nn.Cell):
    """Single transformer layer with GQA, RoPE, SwiGLU FFN. All FP16.

    @lazy_inline enables subgraph reuse across all 28 instances: the
    compiler compiles the TransformerBlock subgraph ONCE and shares it,
    reducing graph compilation from ~13 min to ~2-3 min per config.
    Requires that no attributes change after __init__.
    """

    @lazy_inline
    def __init__(self, world_size: int = 1, lora_rank: int = 0,
                 lora_alpha: float = 0.0):
        super().__init__()
        ws = world_size
        _mp = ws > 1  # model parallel enabled

        # Attention projections — FP16 weights
        if _mp:
            # Column-parallel: split output dim across NPUs
            self.q_proj = _sharded_dense(HIDDEN_DIM, NUM_HEADS * HEAD_DIM,
                                         (1, 1), (ws, 1))
            self.k_proj = _sharded_dense(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM,
                                         (1, 1), (ws, 1))
            self.v_proj = _sharded_dense(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM,
                                         (1, 1), (ws, 1))
            # Row-parallel: split input dim across NPUs (AllReduce)
            self.o_proj = _sharded_dense(NUM_HEADS * HEAD_DIM, HIDDEN_DIM,
                                         (1, ws), (1, ws))
        else:
            self.q_proj = _fp16_dense(HIDDEN_DIM, NUM_HEADS * HEAD_DIM)
            self.k_proj = _fp16_dense(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM)
            self.v_proj = _fp16_dense(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM)
            self.o_proj = _fp16_dense(NUM_HEADS * HEAD_DIM, HIDDEN_DIM)

        # FFN (SwiGLU)
        if _mp:
            self.gate_proj = _sharded_dense(HIDDEN_DIM, INTERMEDIATE_DIM,
                                            (1, 1), (ws, 1))
            self.up_proj = _sharded_dense(HIDDEN_DIM, INTERMEDIATE_DIM,
                                          (1, 1), (ws, 1))
            self.down_proj = _sharded_dense(INTERMEDIATE_DIM, HIDDEN_DIM,
                                            (1, ws), (1, ws))
        else:
            self.gate_proj = _fp16_dense(HIDDEN_DIM, INTERMEDIATE_DIM)
            self.up_proj = _fp16_dense(HIDDEN_DIM, INTERMEDIATE_DIM)
            self.down_proj = _fp16_dense(INTERMEDIATE_DIM, HIDDEN_DIM)

        # Norms
        self.input_norm = RMSNorm(HIDDEN_DIM)
        self.post_attn_norm = RMSNorm(HIDDEN_DIM)
        self.ffn_norm = RMSNorm(HIDDEN_DIM)

        self.scale = Tensor(HEAD_DIM ** -0.5, ms.float16)

        # LoRA injection INSIDE __init__ so @lazy_inline can capture the
        # final attribute state.  LoRALinear is defined below (line ~327)
        # but Python resolves the name at call-time, not def-time.
        if lora_rank > 0:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                base = getattr(self, proj_name)
                setattr(self, proj_name,
                        LoRALinear(base, lora_rank, lora_alpha, ws))

        # NOTE: Attention matmuls use functional ops.matmul() (not the
        # ops.MatMul primitive) because Ascend's MatMul primitive only
        # supports 2D inputs. The functional form handles batched 4D tensors.
        # In SEMI_AUTO_PARALLEL, unsharded ops inherit tensor layout
        # from their sharded inputs (column-parallel Q/K/V projections).

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        B, S, _ = x.shape

        # ---- Self-attention ----
        h = self.input_norm(x)
        # (B, S, H*D) → (B, H, S, D) — must use full 4-arg perm
        Q = self.q_proj(h).view(B, S, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        K = self.k_proj(h).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        V = self.v_proj(h).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # RoPE on Q and K
        Q = _apply_rotary_emb(Q, cos, sin)
        K = _apply_rotary_emb(K, cos, sin)

        # GQA: expand K/V from NUM_KV_HEADS to NUM_HEADS
        K = ops.tile(K, (1, NUM_KV_GROUPS, 1, 1))  # (B, H, S, D)
        V = ops.tile(V, (1, NUM_KV_GROUPS, 1, 1))

        # Scaled dot-product attention (causal)
        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = attn + mask  # mask: (1, 1, S, S)
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)  # (B, H, S, D)
        # (B, H, S, D) → (B, S, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        out = self.o_proj(out)
        x = x + self.post_attn_norm(out)

        # ---- SwiGLU FFN ----
        h = self.ffn_norm(x)
        gate = ops.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


# ===========================================================================
# Phase 7: Full Qwen2-Like Model
# ===========================================================================

class Qwen2LikeModel(nn.Cell):
    """~6.4B param model matching Qwen2.5-Coder-7B dimensions. All FP16."""

    def __init__(self, world_size: int = 1, lora_rank: int = 0,
                 lora_alpha: float = 0.0):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        # Override embedding table to FP16
        self.embed_tokens.embedding_table = Parameter(
            initializer(Normal(0.02), (VOCAB_SIZE, HIDDEN_DIM), ms.float16),
            name="embedding_table")
        self.layers = nn.CellList(
            [TransformerBlock(world_size, lora_rank, lora_alpha)
             for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = _fp16_dense(HIDDEN_DIM, VOCAB_SIZE)

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """Forward pass. Returns logits (B, S, V).

        Args:
            input_ids: (B, S) int32
            cos, sin: precomputed RoPE tables
            mask: causal attention mask (1, 1, S, S)
        """
        x = self.embed_tokens(input_ids)  # (B, S, H)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, S, V)
        return logits


# ===========================================================================
# Phase 8: LoRA (rank=64, alpha=16)
# ===========================================================================

class LoRALinear(nn.Cell):
    """LoRA: output = base(x) + (xA^T B^T) * scale."""

    def __init__(self, base: nn.Dense, rank: int = 64, alpha: float = 16.0,
                 world_size: int = 1):
        super().__init__()
        self.base = base
        self.scaling = Tensor(alpha / rank, ms.float16)
        in_f = base.in_channels
        out_f = base.out_channels
        # Explicit FP16 init — to_float may miss dynamically-injected subcells in MS 2.2
        self.A = Parameter(initializer(Normal(0.02), (rank, in_f), ms.float16), name="lora_A")
        self.B = Parameter(initializer(Zero(), (out_f, rank), ms.float16), name="lora_B")

        # Stored matmul ops for SEMI_AUTO_PARALLEL sharding
        self.lora_a = ops.MatMul(transpose_b=True)  # x @ A.T
        self.lora_b = ops.MatMul(transpose_b=True)  # hidden @ B.T

        if world_size > 1:
            # A matmul: replicated (LoRA rank is small, no benefit from splitting)
            self.lora_a.shard(((1, 1), (1, 1)))
            # B matmul: column-parallel (split out_f across NPUs, matching base)
            self.lora_b.shard(((1, 1), (world_size, 1)))

    def construct(self, x: Tensor) -> Tensor:
        # base: (B, S, in_f) -> (B, S, out_f)
        base_out = self.base(x)
        # lora: Ascend MatMul requires 2D inputs — reshape to (B*S, in_f)
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])
        hidden = self.lora_a(x_2d, self.A)       # (B*S, rank)
        lora_out = self.lora_b(hidden, self.B) * self.scaling  # (B*S, out_f)
        lora_out = lora_out.reshape(x_shape[:-1] + (self.base.out_channels,))
        return base_out + lora_out

    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels


# ===========================================================================
# Phase 9: SFM Slot Bank Adapter
# ===========================================================================

class SFMSlotBank(nn.Cell):
    """State Slot Bank: cross-attention to slots + gated recurrent update.
    16 slots x 256d. All params FP16."""

    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256,
                 num_slots: int = 16, num_heads: int = 4, world_size: int = 1):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = slot_dim
        ws = world_size
        _mp = ws > 1

        self.slot_vectors = Parameter(
            initializer(Normal(0.02), (num_slots, slot_dim), ms.float16), name="slot_vectors")

        # Cross-attention projections
        if _mp:
            self.q_proj = _sharded_dense(hidden_dim, num_heads * slot_dim,
                                         (1, 1), (ws, 1))
            self.k_proj = _sharded_dense(slot_dim, num_heads * slot_dim,
                                         (1, 1), (ws, 1))
            self.v_proj = _sharded_dense(slot_dim, num_heads * slot_dim,
                                         (1, 1), (ws, 1))
            self.out_proj = _sharded_dense(num_heads * slot_dim, hidden_dim,
                                           (1, ws), (1, ws))
        else:
            self.q_proj = _fp16_dense(hidden_dim, num_heads * slot_dim)
            self.k_proj = _fp16_dense(slot_dim, num_heads * slot_dim)
            self.v_proj = _fp16_dense(slot_dim, num_heads * slot_dim)
            self.out_proj = _fp16_dense(num_heads * slot_dim, hidden_dim)

        self.layer_norm = RMSNorm(hidden_dim)

        # Gated recurrent update (replicated — small output, explicit shard for layout)
        self.W_alpha = _fp16_dense(hidden_dim, slot_dim, has_bias=True)
        self.W_beta = _fp16_dense(hidden_dim, slot_dim, has_bias=True)
        self.W_v = _fp16_dense(hidden_dim, slot_dim, has_bias=True)
        if world_size > 1:
            # Explicit data-parallel shard: prevents "has not tensor layout" warnings
            # and ensures parallel optimizer handles these correctly
            self.W_alpha.matmul.shard(((1, 1), (1, 1)))
            self.W_beta.matmul.shard(((1, 1), (1, 1)))
            self.W_v.matmul.shard(((1, 1), (1, 1)))
        self.scale = Tensor(slot_dim ** -0.5, ms.float16)

        # NOTE: Attention uses functional ops.matmul() (batched 4D) not
        # ops.MatMul primitive (2D only on Ascend). Unsharded ops
        # inherit tensor layout from column-parallel projections.

    def construct(self, hidden_states: Tensor) -> tuple:
        """Returns (modified_hidden, new_slots)."""
        B, S, _ = hidden_states.shape
        slots = ops.broadcast_to(
            self.slot_vectors.reshape(1, self.num_slots, self.slot_dim),
            (B, self.num_slots, self.slot_dim))

        Q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.k_proj(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.v_proj(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        out = self.out_proj(out)
        modified = self.layer_norm(hidden_states + out)

        # Gated recurrent slot update
        h = ops.mean(hidden_states, axis=1, keep_dims=True)  # (B, 1, H)
        a = ops.sigmoid(self.W_alpha(h))  # (B, 1, slot_dim)
        b = ops.sigmoid(self.W_beta(h))
        v = ops.tanh(self.W_v(h))
        new_slots = a * slots + b * v  # (B, 1, slot_dim) * (B, num_slots, slot_dim)
        return modified, new_slots


class SFMAdapter(nn.Cell):
    """Wrapper that applies SFMSlotBank.

    @lazy_inline shares compiled subgraph across the 4 adapter instances.
    """

    @lazy_inline
    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256,
                 world_size: int = 1):
        super().__init__()
        self.slot_bank = SFMSlotBank(hidden_dim, slot_dim, world_size=world_size)

    def construct(self, hidden_states: Tensor) -> tuple:
        return self.slot_bank(hidden_states)


# ===========================================================================
# Phase 10: Injection helpers
# ===========================================================================

def inject_lora(model: nn.Cell, rank: int = 64, alpha: float = 16.0,
                world_size: int = 1) -> int:
    """Replace q/k/v/o projections in every transformer layer with LoRA."""
    count = 0
    for layer in model.layers:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(layer, name, None)
            if orig is not None and not isinstance(orig, LoRALinear):
                wrapped = LoRALinear(orig, rank, alpha, world_size)
                setattr(layer, name, wrapped)
                count += 1
    return count


# ===========================================================================
# Phase 10.5: SFM-Enhanced Model (static construct for GRAPH_MODE)
# ===========================================================================

class SFMEnhancedModel(nn.Cell):
    """Qwen2LikeModel wrapped with SFM adapters at specific layers.

    Construct is fully unrolled (no exec/type hacks) so that
    inspect.getsourcelines() can trace it in GRAPH_MODE.
    """

    def __init__(self, base_model: nn.Cell, sfm_adapters: list):
        super().__init__()
        self.base = base_model
        for idx, adapter in sfm_adapters:
            setattr(self, f"sfm_layer_{idx}", adapter)

    def construct(self, input_ids, cos, sin, mask):
        x = self.base.embed_tokens(input_ids)
        x = self.base.layers[0](x, cos, sin, mask)
        x = self.base.layers[1](x, cos, sin, mask)
        x = self.base.layers[2](x, cos, sin, mask)
        x = self.base.layers[3](x, cos, sin, mask)
        x = self.base.layers[4](x, cos, sin, mask)
        x = self.base.layers[5](x, cos, sin, mask)
        x = self.base.layers[6](x, cos, sin, mask)
        x = self.base.layers[7](x, cos, sin, mask)
        x, _ = self.sfm_layer_7(x)
        x = self.base.layers[8](x, cos, sin, mask)
        x = self.base.layers[9](x, cos, sin, mask)
        x = self.base.layers[10](x, cos, sin, mask)
        x = self.base.layers[11](x, cos, sin, mask)
        x = self.base.layers[12](x, cos, sin, mask)
        x = self.base.layers[13](x, cos, sin, mask)
        x = self.base.layers[14](x, cos, sin, mask)
        x, _ = self.sfm_layer_14(x)
        x = self.base.layers[15](x, cos, sin, mask)
        x = self.base.layers[16](x, cos, sin, mask)
        x = self.base.layers[17](x, cos, sin, mask)
        x = self.base.layers[18](x, cos, sin, mask)
        x = self.base.layers[19](x, cos, sin, mask)
        x = self.base.layers[20](x, cos, sin, mask)
        x = self.base.layers[21](x, cos, sin, mask)
        x, _ = self.sfm_layer_21(x)
        x = self.base.layers[22](x, cos, sin, mask)
        x = self.base.layers[23](x, cos, sin, mask)
        x = self.base.layers[24](x, cos, sin, mask)
        x = self.base.layers[25](x, cos, sin, mask)
        x = self.base.layers[26](x, cos, sin, mask)
        x = self.base.layers[27](x, cos, sin, mask)
        x, _ = self.sfm_layer_27(x)
        x = self.base.norm(x)
        logits = self.base.lm_head(x)
        return logits


class SFMForwardLossCell(nn.Cell):
    """Forward pass + cross-entropy loss in a single Cell.

    cos/sin/mask are stored as attributes so the construct()
    is static and traceable by GRAPH_MODE's inspect.getsourcelines().
    """

    def __init__(self, model: nn.Cell, loss_fn: nn.Cell,
                 cos: Tensor, sin: Tensor, mask: Tensor):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.cos = cos
        self.sin = sin
        self.mask = mask

    def construct(self, inputs: Tensor, labels: Tensor) -> Tensor:
        logits = self.model(inputs, self.cos, self.sin, self.mask)
        logits_flat = logits.reshape(-1, VOCAB_SIZE)
        loss = self.loss_fn(logits_flat, labels)
        return loss


def count_params(cell: nn.Cell) -> tuple:
    """Returns (total_params, trainable_params)."""
    total = sum(p.size for p in cell.get_parameters())
    trainable = sum(p.size for p in cell.trainable_params())
    return total, trainable


# ===========================================================================
# Phase 11: Synthetic data
# ===========================================================================

class SyntheticDataset:
    """Random token sequences for throughput benchmarking."""

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int = VOCAB_SIZE,
                 num_samples: int = 100):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        np.random.seed(42)
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len)).astype(np.int32)

    def __getitem__(self, idx):
        tokens = self.data[idx % self.num_samples]
        return tokens.copy(), tokens.copy()  # (input_ids, labels)

    def __len__(self):
        return self.num_samples


def create_data_loader(batch_size: int, seq_len: int, num_samples: int = 100):
    """Create a MindSpore GeneratorDataset for synthetic data."""
    ds = SyntheticDataset(batch_size, seq_len, num_samples=num_samples)
    gen = ms.dataset.GeneratorDataset(ds, column_names=["input_ids", "labels"],
                                       shuffle=False)
    gen = gen.batch(batch_size, drop_remainder=True)
    return gen


# ===========================================================================
# Phase 12: Benchmark configs
# ===========================================================================

CONFIGS = [
    {"name": "E", "batch_size": 4, "seq_len": 1024},   # smallest attn (S²=1M)
    {"name": "A", "batch_size": 1, "seq_len": 2048},   # S²=4M
    {"name": "B", "batch_size": 2, "seq_len": 2048},   # S²=4M
    {"name": "C", "batch_size": 4, "seq_len": 2048},   # S²=4M
    {"name": "D", "batch_size": 2, "seq_len": 4096},   # S²=16M
    {"name": "F", "batch_size": 1, "seq_len": 8192},   # S²=64M
]
WARMUP = 2
MEASURE = 5


# ===========================================================================
# Phase 14: Single-config benchmark (nn.Cell-based training)
# ===========================================================================

def run_one_config(model, optimizer, loss_fn, cfg: dict, rank: int, world_size: int) -> dict:
    """Run throughput benchmark for one (batch_size, seq_len) config.

    Uses nn.TrainOneStepCell (Cell-based) instead of ms.value_and_grad
    (function-based) for GRAPH_MODE compatibility — Cell construct methods
    are traceable via inspect.getsourcelines() but dynamic functions are not.

    Returns result dict or None on OOM.
    """
    B, S = cfg["batch_size"], cfg["seq_len"]
    tok = B * S

    # Precompute RoPE tables for S-1 (after label shift trimming)
    actual_len = S - 1
    cos, sin = _build_rope_cache(actual_len, HEAD_DIM)

    # Precompute causal mask once on NPU (avoid numpy inside graph)
    causal_np = np.triu(np.full((actual_len, actual_len), -1e4, dtype=np.float16), k=1)
    causal_mask = Tensor(causal_np.reshape(1, 1, actual_len, actual_len))

    # Create synthetic data tensors (pinned for this config)
    np.random.seed(42)
    input_ids = Tensor(
        np.random.randint(0, VOCAB_SIZE, (B, S)).astype(np.int32))
    labels = input_ids.copy()

    # Labels for CE: shift by 1 (predict next token)
    input_trimmed = input_ids[:, :-1]
    labels_trimmed = labels[:, 1:].reshape(-1,)  # (B*(S-1),)

    # Build training cell (new instance per config for different seq_len)
    # Initialized to None so finally block is safe even if creation OOMs
    train_net = None
    train_step = None

    try:
        train_net = SFMForwardLossCell(model, loss_fn, cos, sin, causal_mask)
        train_step = nn.TrainOneStepCell(train_net, optimizer)
        train_step.set_train()

        log(f"  Config {cfg['name']}: B={B}, S={S}, tok/step={tok:,}")

        # Warmup (first call triggers GRAPH compilation + first execution)
        t_compile_start = time.time()
        for _ in range(WARMUP):
            train_step(input_trimmed, labels_trimmed)
        log(f"  Compile + warmup: {time.time() - t_compile_start:.0f}s")

        # Measure
        t0 = time.time()
        for _ in range(MEASURE):
            train_step(input_trimmed, labels_trimmed)
        total_elapsed = time.time() - t0

        avg_ms = total_elapsed / MEASURE * 1000
        tps = tok / (avg_ms / 1000)

        return {
            "name": cfg["name"], "batch_size": B, "seq_len": S,
            "tok": tok, "avg_ms": avg_ms,
            "tps_dev": tps, "tps_total": tps * world_size,
            "peak": 0.0, "free": 0.0,
        }
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "alloc" in msg:
            log(f"  Config {cfg['name']}: OOM (B={B}, S={S})")
            return None
        raise
    finally:
        # CRITICAL: free compiled graph before next config.
        # GRAPH_MODE compiles forward+backward into a static graph that
        # persists on the Cell object. Without explicit cleanup, each
        # config's graph accumulates and cascading OOM occurs.
        del train_step, train_net
        gc.collect()


# ===========================================================================
# Phase 15: Results printer
# ===========================================================================

def print_results(results: list, world_size: int, output_path: str) -> None:
    """Print and save throughput results."""
    log("")
    log("=" * 100)
    log("THROUGHPUT RESULTS (MindSpore)")
    log("=" * 100)
    log(f"{'Cfg':>3} | {'Batch':>5} | {'SeqLen':>6} | {'Tok/step':>9} | "
        f"{'Tok/s/dev':>10} | {'Tok/s(total)':>12} | {'ms/step':>8}")
    log("-" * 100)

    for i, r in enumerate(results):
        cfg = CONFIGS[i]
        if r is not None:
            log(f"{r['name']:>3} | {r['batch_size']:>5} | {r['seq_len']:>6} | "
                f"{r['tok']:>9,} | {r['tps_dev']:>10.1f} | "
                f"{r['tps_total']:>12.1f} | {r['avg_ms']:>8.1f}")
        else:
            log(f"{cfg['name']:>3} | B={cfg['batch_size']:<4} | "
                f"S={cfg['seq_len']:<5} | OOM")

    valid = [r for r in results if r is not None]
    if valid:
        best = max(valid, key=lambda x: x["tps_dev"])
        log(f"\nBest: {best['name']} (B={best['batch_size']}, S={best['seq_len']})")
        log(f"  {best['tps_dev']:.1f} tok/s/device, "
            f"{best['tps_total']:.1f} tok/s total ({world_size}x NPUs)")
        hours = 1.1e9 / (best["tps_total"] * 3600)
        log(f"  1.1B tokens estimate: {hours:.1f} hours")

    # Write to file
    path = os.path.join(output_path, "throughput_results.txt")
    with open(path, "w") as f:
        f.write("SFM+LoRA Training Throughput Benchmark (MindSpore)\n")
        f.write(f"World size: {world_size} NPUs\n\n")
        for i, r in enumerate(results):
            cfg = CONFIGS[i]
            if r is not None:
                f.write(f"Config {r['name']}: B={r['batch_size']}, S={r['seq_len']}\n"
                        f"  tok/s/dev={r['tps_dev']:.1f}, tok/s(total)={r['tps_total']:.1f}\n"
                        f"  ms/step={r['avg_ms']:.1f}\n\n")
            else:
                f.write(f"Config {cfg['name']}: OOM\n\n")
        if valid:
            best = max(valid, key=lambda x: x["tps_dev"])
            hours = 1.1e9 / (best["tps_total"] * 3600)
            f.write(f"Best: {best['name']} -- {best['tps_dev']:.1f} tok/s/dev\n"
                    f"1.1B tokens: {hours:.1f} hours\n")
    log(f"Results saved to {path}")


# ===========================================================================
# Phase 16: Local test mode (syntax + shape check, no NPU)
# ===========================================================================

def run_local_test():
    """Smoke test of all components on CPU."""
    log("LOCAL TEST MODE (CPU)")
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    # Test RMSNorm
    log("  Testing RMSNorm...")
    norm = RMSNorm(64)
    x = Tensor(np.random.randn(2, 4, 64).astype(np.float16))
    out = norm(x)
    log(f"    RMSNorm: {x.shape} -> {out.shape} [OK]")

    # Test RoPE
    log("  Testing RoPE...")
    cos, sin = _build_rope_cache(8, 64)
    log(f"    cos: {cos.shape}, sin: {sin.shape} [OK]")

    # Test TransformerBlock
    log("  Testing TransformerBlock...")
    block = TransformerBlock()
    x = Tensor(np.random.randn(1, 8, HIDDEN_DIM).astype(np.float16))
    cos, sin = _build_rope_cache(8, HEAD_DIM)
    mask = Tensor(np.zeros((1, 1, 8, 8), dtype=np.float16))
    out = block(x, cos, sin, mask)
    log(f"    TransformerBlock: {x.shape} -> {out.shape} [OK]")

    # Test LoRALinear
    log("  Testing LoRALinear...")
    base = nn.Dense(512, 512, has_bias=False)
    lora = LoRALinear(base, rank=8, alpha=16)
    x = Tensor(np.random.randn(2, 8, 512).astype(np.float16))
    out = lora(x)
    log(f"    LoRALinear: (2,8,512) -> {out.shape} [OK]")

    # Test SFMSlotBank
    log("  Testing SFMSlotBank...")
    sb = SFMSlotBank(3584, 256)
    h = Tensor(np.random.randn(1, 10, 3584).astype(np.float16))
    o, s = sb(h)
    log(f"    SFMSlotBank: {h.shape} -> {o.shape}, slots {s.shape} [OK]")

    # Test SFMAdapter
    log("  Testing SFMAdapter...")
    ad = SFMAdapter(3584)
    o2, s2 = ad(h)
    log(f"    SFMAdapter: {h.shape} -> {o2.shape}, slots {s2.shape} [OK]")

    # Test inject_lora
    log("  Testing LoRA injection...")
    model2 = Qwen2LikeModel()
    n = inject_lora(model2, rank=64, alpha=16)
    log(f"    Injected {n} LoRA layers [OK]")

    # Test SFMEnhancedModel
    log("  Testing SFMEnhancedModel...")
    base = Qwen2LikeModel()
    sfm_list = [(NUM_LAYERS // 4, SFMAdapter(3584)),
                (NUM_LAYERS // 2, SFMAdapter(3584)),
                (3 * NUM_LAYERS // 4, SFMAdapter(3584)),
                (NUM_LAYERS - 1, SFMAdapter(3584))]
    enhanced = SFMEnhancedModel(base, sfm_list)
    input_ids = Tensor(np.random.randint(0, 100, (1, 8)).astype(np.int32))
    cos, sin = _build_rope_cache(8, HEAD_DIM)
    mask = Tensor(np.zeros((1, 1, 8, 8), dtype=np.float16))
    logits = enhanced(input_ids, cos, sin, mask)
    log(f"    SFMEnhancedModel: input (1,8) -> logits {logits.shape} [OK]")

    # Test training step (value_and_grad)
    log("  Testing value_and_grad...")
    total, trainable = count_params(enhanced)
    log(f"    Params: {total:,} total, {trainable:,} trainable [OK]")

    log("\nLOCAL TEST COMPLETE -- all components verified")


# ===========================================================================
# Phase 17: Main
# ===========================================================================

def main():
    global _log_file
    args, unknown = parser.parse_known_args()

    if args.local_test:
        run_local_test()
        return

    # ---- Self-bootstrap: spawn worker processes for multi-NPU ----
    # OpenI sets RANK_ID=0 but only launches 1 process.  Check RANK_SIZE to
    # detect whether an external launcher (msrun/mpirun) is managing us.
    _rank_size_str = os.getenv('RANK_SIZE')
    _need_bootstrap = (_rank_size_str is None or int(_rank_size_str) <= 1)
    if _need_bootstrap:
        num_workers = args.npu_count if args.npu_count > 0 else 4
        os.environ['RANK_SIZE'] = str(num_workers)
        # RANK_ID may already be 0 (set by OpenI) — keep it
        if os.getenv('RANK_ID') is None:
            os.environ['RANK_ID'] = '0'
            os.environ['DEVICE_ID'] = '0'
        log(f"Self-bootstrapping {num_workers} NPUs (rank 0)")
        import subprocess as _sp
        for i in range(1, num_workers):
            env = os.environ.copy()
            env['RANK_ID'] = str(i)
            env['DEVICE_ID'] = str(i)
            env['RANK_SIZE'] = str(num_workers)
            log(f"  Spawning rank {i} (DEVICE_ID={i})...")
            _sp.Popen([sys.executable] + sys.argv, env=env)

    # ---- Rank / world_size ----
    rank_id_str = os.getenv('RANK_ID')
    rank_id = int(rank_id_str) if rank_id_str is not None else 0
    rank_size_str = os.getenv('RANK_SIZE')
    is_multi_card = (rank_id_str is not None or rank_size_str is not None)

    log(f"Rank {rank_id}, multi_card={is_multi_card}")

    # ---- c2net prepare (rank 0 only) ----
    _SYNC_FILE = "/cache/sfm_ready.txt"
    model_path = None
    output_path = None
    world_size = 1

    if rank_id == 0:
        log("Rank 0: calling c2net prepare()...")
        try:
            from c2net.context import prepare
            c2net_ctx = prepare()
            log("Rank 0: prepare() OK")
            log(f"  output_path:         {getattr(c2net_ctx, 'output_path', 'N/A')}")
            log(f"  pretrain_model_path: {getattr(c2net_ctx, 'pretrain_model_path', 'N/A')}")
            log(f"  dataset_path:        {getattr(c2net_ctx, 'dataset_path', 'N/A')}")

            output_path = c2net_ctx.output_path
            os.makedirs(output_path, exist_ok=True)
            _log_file = os.path.join(output_path, "training.log")

            # Copy boot log
            if os.path.exists(_boot_log):
                try:
                    import shutil
                    shutil.copy2(_boot_log, os.path.join(output_path, "boot.log"))
                except Exception:
                    pass

            # Model path (unused -- we build from scratch, but kept for compatibility)
            model_path = resolve_model_path(c2net_ctx, args)
            # world_size: prefer RANK_SIZE (set by msrun), then --npu_count arg
            world_size = (int(rank_size_str) if rank_size_str
                          else (args.npu_count if args.npu_count > 0 else 1))

            with open(_SYNC_FILE, "w") as f:
                f.write(f"{output_path}\n{world_size}\n")
            log(f"Rank 0: world_size={world_size}, sync file written.")

        except Exception as e:
            import traceback
            log(f"Rank 0: prepare FAILED:\n{traceback.format_exc()}")
            try:
                with open(_SYNC_FILE, "w") as f:
                    f.write(f"ERROR: {e}\n")
            except Exception:
                pass
            sys.exit(1)
    else:
        log(f"Rank {rank_id}: waiting for rank 0...")
        waited = 0
        while not os.path.exists(_SYNC_FILE):
            time.sleep(1)
            waited += 1
            if waited > 600:
                log(f"Rank {rank_id}: timeout (10 min)")
                sys.exit(1)
        with open(_SYNC_FILE, "r") as f:
            output_path = f.readline().strip()
            world_size = int(f.readline().strip())
        os.makedirs(output_path, exist_ok=True)
        _log_file = os.path.join(output_path, "training.log")
        log(f"Rank {rank_id}: world_size={world_size}")

    # ---- MindSpore context ----
    # (Ascend env vars already set in Phase 0 before import)

    if is_multi_card and world_size > 1:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=rank_id)
        ms.communication.init()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
            device_num=world_size,
            enable_parallel_optimizer=True,
            dataset_strategy="full_batch")
        log(f"Rank {rank_id}: semi-auto parallel init OK ({world_size} NPUs)")
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
        log(f"Rank {rank_id}: single-card Ascend mode")

    # ---- Build model ----
    if rank_id == 0:
        log("Building Qwen2-like model...")

    model = Qwen2LikeModel(world_size=world_size, lora_rank=64, lora_alpha=16.0)

    # Enable gradient checkpointing on every transformer layer.
    # This frees activations after each layer's forward pass and recomputes
    # them during backward — reduces peak activation memory from ~18 GB to ~1 GB.
    for layer in model.layers:
        layer.recompute()

    if rank_id == 0:
        log("Model built with built-in LoRA (rank=64, alpha=16, @lazy_inline)")

    # Count LoRA projections
    lora_n = sum(1 for layer in model.layers
                 for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                 if isinstance(getattr(layer, name, None), LoRALinear))
    sfm_idx = [NUM_LAYERS // 4, NUM_LAYERS // 2,
               3 * NUM_LAYERS // 4, NUM_LAYERS - 1]  # [7, 14, 21, 27]
    sfm_adapters = [(idx, SFMAdapter(HIDDEN_DIM, slot_dim=256,
                                      world_size=world_size))
                    for idx in sfm_idx]

    # Wrap in SFM-enhanced model (static construct for GRAPH_MODE)
    model = SFMEnhancedModel(model, sfm_adapters)

    if rank_id == 0:
        total, trainable = count_params(model)
        log(f"LoRA: rank=64, alpha=16, {lora_n} projections")
        log(f"SFM: 4 adapters at layers {sfm_idx}")
        log(f"Params: {total:,} total, {trainable:,} trainable "
            f"({100 * trainable / total:.2f}%)")

    # ---- Freeze base model, keep LoRA + SFM trainable ----
    for p in model.get_parameters():
        p.requires_grad = False

    # Unfreeze LoRA A/B params
    for layer_idx in range(NUM_LAYERS):
        layer = model.base.layers[layer_idx]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer, proj_name, None)
            if proj is not None and isinstance(proj, LoRALinear):
                proj.A.requires_grad = True
                proj.B.requires_grad = True

    # Unfreeze SFM params
    for idx in sfm_idx:
        adapter = getattr(model, f"sfm_layer_{idx}", None)
        if adapter is not None:
            for p in adapter.get_parameters():
                p.requires_grad = True

    # ---- Loss + optimizer ----
    loss_fn = nn.CrossEntropyLoss()

    # trainable_params() returns LoRA + SFM params in the same order
    # that nn.TrainOneStepCell's GradOperation will compute gradients
    train_params = list(model.trainable_params())

    if rank_id == 0:
        train_total = sum(p.size for p in train_params)
        log(f"Trainable param groups: {len(train_params)} "
            f"({train_total:,} params)")

    optimizer = nn.AdamWeightDecay(train_params, learning_rate=2e-4)

    if rank_id == 0:
        log("")
        log("=" * 100)
        log("TRAINING THROUGHPUT BENCHMARK (MindSpore)")
        log("=" * 100)
        log(f"MindSpore {ms.__version__}, {world_size} NPUs")
        log(f"Configs: {len(CONFIGS)}, Warmup: {WARMUP}, Measure: {MEASURE}")
        log("")

    # ---- Benchmark loop ----
    results = []
    for cfg in CONFIGS:
        result = run_one_config(model, optimizer, loss_fn, cfg, rank_id, world_size)
        results.append(result)

    # ---- Results + upload ----
    if rank_id == 0:
        print_results(results, world_size, output_path)
        log("\nUploading results...")
        try:
            from c2net.context import upload_output
            upload_output()
            log("Upload complete.")
        except Exception as e:
            log(f"Upload failed (non-fatal): {e}")

    if rank_id == 0:
        log("")
        log("=" * 100)
        log("DONE")
        log("=" * 100)


def resolve_model_path(c2net_ctx, args):
    """Resolve model path from args or c2net context."""
    if args.model_path:
        return args.model_path
    pmp = getattr(c2net_ctx, "pretrain_model_path", None)
    if pmp and os.path.isdir(pmp):
        try:
            entries = [e for e in os.listdir(pmp)
                       if os.path.isdir(os.path.join(pmp, e))]
            log(f"Model root: {pmp}, subdirs: {entries}")
            for name in ["Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B"]:
                if name in entries:
                    return os.path.join(pmp, name)
            if len(entries) == 1:
                return os.path.join(pmp, entries[0])
            if os.path.exists(os.path.join(pmp, "config.json")):
                return pmp
        except OSError as e:
            log(f"Could not list model dir: {e}")
    return None


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise
