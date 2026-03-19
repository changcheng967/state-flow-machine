"""train.py — Thinker-1.5B v2 from-scratch training with SFM.

v2 changes:
  - Real GRU with reset gate for SFM slot bank
  - Slot persistence across all 4 SFM layers
  - Standard pre-norm TransformerBlock (removed post_attn_norm)
  - Slot prediction via CE loss over 512-token slot vocabulary
  - Initial slot vectors as learned parameter
  - Multi-turn training with [TURN] separator
  - FP16 params (FP32 master weights via AdamWeightDecay)

Self-contained MindSpore 2.2 training script for OpenI (4x Ascend 910).
All model classes are INLINED (not imported) because MS 2.2's
inspect.getsourcelines() must trace within the same file for GRAPH_MODE.
"""

import os
import sys
import json
import time
import math
import gc
import glob
import warnings
import struct
import random

try:
    _boot_ts = time.strftime("%H:%M:%S")
    _boot_pid = os.getpid()
    _boot_rank = os.environ.get("RANK_ID", "?")
    sys.stderr.write(
        f"[BOOT {_boot_ts}] pid={_boot_pid} rank={_boot_rank} started\n")
    sys.stderr.flush()
    os.makedirs("/cache/output", exist_ok=True)
    with open("/cache/output/boot.log", "a") as _bf:
        _bf.write(
            f"[{_boot_ts}] pid={_boot_pid} rank={_boot_rank} "
            f"thinker15b_v2_started\n")
except Exception:
    pass

import subprocess
warnings.filterwarnings("ignore")

os.environ.update({
    "MS_COMPILER_CACHE_ENABLE": "1",
    "MS_COMPILER_CACHE_PATH": "/home/ma-user/work/graph_cache",
    "MS_BUILD_PROCESS_NUM": "24",
    "TASK_QUEUE_ENABLE": "2",
    "CPU_AFFINITY_CONF": "1",
    "ASCEND_GLOBAL_LOG_LEVEL": "3",
    "GLOG_v": "2",
    "HCCL_CONNECT_TIMEOUT": "1800",
    "MS_COMPILER_OP_LEVEL": "0",
})

# ── Paths (c2net + local fallback) ──────────────────────────────────
_PARENT_PASSED = os.environ.get("SFM_OUTPUT_PATH") is not None
HAS_C2NET = False
CODE_PATH = "/home/ma-user/work/code"
DATASET_PATH = "/home/ma-user/work/dataset"
PRETRAIN_MODEL_PATH = "/home/ma-user/work/pretrainmodel"
OUTPUT_PATH = "/home/ma-user/work/output"

if _PARENT_PASSED:
    OUTPUT_PATH = os.environ["SFM_OUTPUT_PATH"]
    DATASET_PATH = os.environ["SFM_DATASET_PATH"]
    PRETRAIN_MODEL_PATH = os.environ["SFM_PRETRAIN_PATH"]
    CODE_PATH = os.environ.get("SFM_CODE_PATH", CODE_PATH)
else:
    try:
        from c2net.context import prepare, upload_output
        _ctx = prepare()
        CODE_PATH = _ctx.code_path
        DATASET_PATH = _ctx.dataset_path
        PRETRAIN_MODEL_PATH = _ctx.pretrain_model_path
        OUTPUT_PATH = _ctx.output_path
        HAS_C2NET = True
        print(f"c2net initialised: code={CODE_PATH}, "
              f"dataset={DATASET_PATH}, output={OUTPUT_PATH}", flush=True)
    except Exception:
        print("c2net not available — using default paths", flush=True)

if not os.path.isdir(CODE_PATH) and os.environ.get("LOCAL_CODE_PATH"):
    CODE_PATH = os.environ["LOCAL_CODE_PATH"]
if not os.path.isdir(DATASET_PATH) and os.environ.get("LOCAL_DATASET_PATH"):
    DATASET_PATH = os.environ["LOCAL_DATASET_PATH"]
if not os.path.isdir(PRETRAIN_MODEL_PATH) and \
        os.environ.get("LOCAL_PRETRAIN_MODEL_PATH"):
    PRETRAIN_MODEL_PATH = os.environ["LOCAL_PRETRAIN_MODEL_PATH"]
if not OUTPUT_PATH or not os.path.isdir(OUTPUT_PATH):
    OUTPUT_PATH = os.environ.get("LOCAL_OUTPUT_PATH", "/cache/output")

CKPT_DIR = os.path.join(OUTPUT_PATH, "checkpoints")

# ── Hyperparameters ─────────────────────────────────────────────────
VOCAB_SIZE = 32000
HIDDEN_DIM = 2048
NUM_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 24
INTERMEDIATE_DIM = 6144
MAX_SEQ_LEN = 4096
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0

# SFM v2
SFM_NUM_SLOTS = 16
SFM_SLOT_DIM = 256
SFM_NUM_HEADS = 4
SFM_LAYERS = (5, 11, 17, 23)
SFM_SLOT_VOCAB_SIZE = 512

# Training
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TIME_LIMIT = 86400
MIN_EPOCHS = 2
CONVERGENCE_WINDOW = 200
CONVERGENCE_PATIENCE = 1000
CONVERGENCE_THRESHOLD = 0.001
SLOT_PRED_WEIGHT = 0.15

# Multi-turn
MAX_TURNS = 4

# Curriculum
CURRICULUM_P3_START = 0.30
CURRICULUM_P4_START = 0.60

RANK_SIZE = 4


# ── Logging ─────────────────────────────────────────────────────────
_log_fh = None


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def setup_logging(rank_id: int) -> None:
    global _log_fh
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_PATH,
                        "train.log" if rank_id == 0
                        else f"worker_{rank_id}.log")
    _log_fh = open(path, "a")
    log(f"Logging to {path}")


def is_main_process() -> bool:
    return os.environ.get("RANK_ID") is None and "--worker" not in sys.argv


def launch_distributed() -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    log("Parent: launching 4 worker processes")
    procs = []
    for i in range(RANK_SIZE):
        env = os.environ.copy()
        env["RANK_ID"] = str(i)
        env["DEVICE_ID"] = str(i)
        env["RANK_SIZE"] = str(RANK_SIZE)
        env["SFM_OUTPUT_PATH"] = OUTPUT_PATH
        env["SFM_DATASET_PATH"] = DATASET_PATH
        env["SFM_PRETRAIN_PATH"] = PRETRAIN_MODEL_PATH
        env["SFM_CODE_PATH"] = CODE_PATH
        log_path = os.path.join(OUTPUT_PATH, f"worker_{i}.log")
        fh = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            env=env, stdout=fh, stderr=subprocess.STDOUT,
        )
        procs.append((p, fh))

    for p, fh in procs:
        p.wait()
        fh.close()
        log(f"Worker (pid={p.pid}) exited with code {p.returncode}")

    results_path = os.path.join(OUTPUT_PATH, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        log("=" * 60)
        log("RESULTS SUMMARY")
        log("=" * 60)
        for k, v in results.items():
            log(f"  {k}: {v}")

    if HAS_C2NET:
        try:
            upload_output()
            log("upload_output() completed successfully")
        except Exception as e:
            log(f"upload_output() failed: {e}")


if is_main_process():
    launch_distributed()
    sys.exit(0)

print(f"[worker] RANK_ID={os.environ.get('RANK_ID')}, "
      f"OUTPUT_PATH={OUTPUT_PATH}, PID={os.getpid()}", flush=True)


# ── MindSpore imports (AFTER env vars) ─────────────────────────────
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Zero, Normal, One
from mindspore.common.tensor import Tensor


# ════════════════════════════════════════════════════════════════════
# INLINE MODEL CLASSES (GRAPH_MODE traceability)
# ════════════════════════════════════════════════════════════════════


def _fp16_dense(in_ch: int, out_ch: int,
                has_bias: bool = False) -> nn.Dense:
    """Create nn.Dense with FP16 parameters."""
    w = np.random.randn(out_ch, in_ch).astype(np.float16) * (
        1.0 / np.sqrt(in_ch))
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

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = Tensor(eps, ms.float32)
        self.weight = ms.Parameter(
            Tensor(np.ones(dim, dtype=np.float32)), name="norm_weight")

    def construct(self, x: Tensor) -> Tensor:
        x_f = x.astype(ms.float32)
        variance = ops.mean(x_f * x_f, axis=-1, keep_dims=True)
        x_norm = x_f * ops.rsqrt(variance + self.eps)
        return (x_norm * self.weight).astype(x.dtype)


class RotaryEmbedding(nn.Cell):
    """Pre-computed RoPE cos/sin tables."""

    def __init__(self, head_dim: int, max_seq: int,
                 theta: float = ROPE_THETA):
        super().__init__()
        inv_freq = 1.0 / (theta ** (
            np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float16)
        sin = np.sin(emb).astype(np.float16)
        self.cos_table = Tensor(cos[np.newaxis, np.newaxis, :, :])
        self.sin_table = Tensor(sin[np.newaxis, np.newaxis, :, :])

    def construct(self, x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = ops.concat([-x2, x1], axis=-1)
        return x * self.cos_table + rotated * self.sin_table


class TransformerBlock(nn.Cell):
    """Single transformer layer: MHA + SwiGLU FFN + RMSNorm (pre-norm only).

    v2: no post_attn_norm. Standard pre-norm: x = x + attn_out.
    """

    def __init__(self) -> None:
        super().__init__()
        H = HIDDEN_DIM
        A = INTERMEDIATE_DIM
        NH = NUM_HEADS
        HD = HEAD_DIM

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

        S = MAX_SEQ_LEN
        mask_np = np.triu(
            np.full((S, S), -1e4, dtype=np.float16), k=1)
        self.causal_mask = Tensor(mask_np[np.newaxis, np.newaxis, :, :])

        self.rope = RotaryEmbedding(HD, S, ROPE_THETA)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> Tensor:
        B, S, _ = x.shape
        NH = NUM_HEADS
        HD = HEAD_DIM

        h = self.input_norm(x)
        Q = self.q_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(h).view(B, S, NH, HD).transpose(0, 2, 1, 3)

        HD2 = HEAD_DIM // 2

        Q = Q * cos + ops.concat([-Q[..., HD2:], Q[..., :HD2]], -1) * sin
        K = K * cos + ops.concat([-K[..., HD2:], K[..., :HD2]], -1) * sin

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
    """GRU-based State Slot Bank with real GRU (v2).

    Real GRU with reset gate:
      z = sigmoid(W_z * [slots; mean_hidden])
      r = sigmoid(W_r * [slots; mean_hidden])
      h_cand = tanh(W_h * [r * slots; mean_hidden])
      new_slots = (1-z) * slots + z * h_cand

    Cross-attention: Q from hidden, K/V from slots.
    Returns (modified_hidden, new_slots).
    """

    def __init__(self) -> None:
        super().__init__()
        H = HIDDEN_DIM
        NS = SFM_NUM_SLOTS
        SD = SFM_SLOT_DIM
        NH = SFM_NUM_HEADS
        self.head_dim = SD // NH

        self.q_proj = _fp16_dense(H, NH * self.head_dim)
        self.k_proj = _fp16_dense(SD, NH * self.head_dim)
        self.v_proj = _fp16_dense(SD, NH * self.head_dim)
        self.out_proj = _fp16_dense(NH * self.head_dim, H)
        self.layer_norm = RMSNorm(H)

        # Real GRU gates
        gru_in = SD + H
        self.W_z = _fp16_dense(gru_in, SD, has_bias=True)
        self.W_r = _fp16_dense(gru_in, SD, has_bias=True)
        self.W_h = _fp16_dense(gru_in, SD, has_bias=True)

        self.scale = Tensor(self.head_dim ** -0.5, ms.float16)

    def construct(self, hidden_states: Tensor, slots: Tensor) -> tuple:
        B, S, _ = hidden_states.shape
        NS = SFM_NUM_SLOTS
        NH = SFM_NUM_HEADS
        HD = self.head_dim

        Q = self.q_proj(hidden_states).view(B, S, NH, HD).transpose(
            0, 2, 1, 3)
        K = self.k_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(slots).view(B, NS, NH, HD).transpose(0, 2, 1, 3)

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        read_content = self.out_proj(out)

        modified = self.layer_norm(hidden_states + read_content)

        # GRU update
        h_mean = ops.mean(hidden_states, axis=1, keep_dims=True)
        h_mean_broad = ops.broadcast_to(
            h_mean, (B, NS, HIDDEN_DIM))

        gru_input = ops.concat([slots, h_mean_broad], axis=-1)

        z = ops.sigmoid(self.W_z(gru_input))
        r = ops.sigmoid(self.W_r(gru_input))
        h_cand = ops.tanh(
            self.W_h(ops.concat([r * slots, h_mean_broad], axis=-1)))
        new_slots = (1.0 - z) * slots + z * h_cand

        return modified, new_slots


class SlotPredictionHead(nn.Cell):
    """Predict slot contents discretized to slot vocabulary (v2 CE loss)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = _fp16_dense(SFM_SLOT_DIM, SFM_SLOT_VOCAB_SIZE)

    def construct(self, slots: Tensor) -> Tensor:
        """Returns (B, num_slots, slot_vocab_size) logits."""
        return self.proj(slots)


class SlotTokenizer(nn.Cell):
    """Discretize slots to nearest slot vocabulary entry."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_vectors = ms.Parameter(
            Tensor(np.random.randn(
                SFM_SLOT_VOCAB_SIZE, SFM_SLOT_DIM).astype(
                    np.float16) * 0.02),
            name="slot_vocab_vectors",
        )

    def construct(self, slots: Tensor) -> Tensor:
        """Returns (B, num_slots) integer IDs."""
        slots_norm = slots / (ops.norm(slots, axis=-1, keep_dims=True) + 1e-8)
        vocab_norm = self.vocab_vectors / (
            ops.norm(self.vocab_vectors, axis=-1, keep_dims=True) + 1e-8)
        sim = ops.matmul(slots_norm, vocab_norm.transpose(1, 0))
        return ops.argmax(sim, axis=-1)


class Thinker15BModel(nn.Cell):
    """Full ~1.45B decoder-only LM with SFM slot banks (v2).

    Construct is UNROLLED for GRAPH_MODE safety.
    Slot persistence across all 4 SFM layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = ms.Parameter(
            Tensor(np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(
                np.float16) * 0.02),
            name="lm_head_weight")
        self.matmul = ops.MatMul(transpose_b=True)

        # Initial slots — learned parameter
        self.initial_slots = ms.Parameter(
            Tensor(np.random.randn(1, SFM_NUM_SLOTS, SFM_SLOT_DIM).astype(
                np.float16) * 0.02),
            name="initial_slots")

        # 24 transformer layers (unrolled)
        self.layer0 = TransformerBlock()
        self.layer1 = TransformerBlock()
        self.layer2 = TransformerBlock()
        self.layer3 = TransformerBlock()
        self.layer4 = TransformerBlock()
        self.layer5 = TransformerBlock()
        self.layer6 = TransformerBlock()
        self.layer7 = TransformerBlock()
        self.layer8 = TransformerBlock()
        self.layer9 = TransformerBlock()
        self.layer10 = TransformerBlock()
        self.layer11 = TransformerBlock()
        self.layer12 = TransformerBlock()
        self.layer13 = TransformerBlock()
        self.layer14 = TransformerBlock()
        self.layer15 = TransformerBlock()
        self.layer16 = TransformerBlock()
        self.layer17 = TransformerBlock()
        self.layer18 = TransformerBlock()
        self.layer19 = TransformerBlock()
        self.layer20 = TransformerBlock()
        self.layer21 = TransformerBlock()
        self.layer22 = TransformerBlock()
        self.layer23 = TransformerBlock()

        # 4 SFM banks at layers 5, 11, 17, 23
        self.sfm5 = SFMSlotBank()
        self.sfm5_pred = SlotPredictionHead()
        self.sfm5_tok = SlotTokenizer()
        self.sfm11 = SFMSlotBank()
        self.sfm11_pred = SlotPredictionHead()
        self.sfm11_tok = SlotTokenizer()
        self.sfm17 = SFMSlotBank()
        self.sfm17_pred = SlotPredictionHead()
        self.sfm17_tok = SlotTokenizer()
        self.sfm23 = SFMSlotBank()
        self.sfm23_pred = SlotPredictionHead()
        self.sfm23_tok = SlotTokenizer()

        self.ce_loss = nn.CrossEntropyLoss()

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> tuple:
        """Returns (logits, total_slot_pred_loss)."""
        B = input_ids.shape[0]
        x = self.embedding(input_ids).astype(ms.float16)
        total_slot_loss = Tensor(0.0, ms.float32)

        # Initialize slots from learned param, broadcast
        slots = ops.broadcast_to(
            self.initial_slots,
            (B, SFM_NUM_SLOTS, SFM_SLOT_DIM))

        # Unrolled layers with slot persistence
        x = self.layer0(x, cos, sin, mask)
        x = self.layer1(x, cos, sin, mask)
        x = self.layer2(x, cos, sin, mask)
        x = self.layer3(x, cos, sin, mask)
        x = self.layer4(x, cos, sin, mask)
        x = self.layer5(x, cos, sin, mask)
        x, slots = self.sfm5(x, slots)
        pred5_logits = self.sfm5_pred(slots)
        slot5_ids = self.sfm5_tok(slots)
        total_slot_loss = total_slot_loss + self.ce_loss(
            pred5_logits.reshape((-1, SFM_SLOT_VOCAB_SIZE)),
            slot5_ids.reshape((-1,)))

        x = self.layer6(x, cos, sin, mask)
        x = self.layer7(x, cos, sin, mask)
        x = self.layer8(x, cos, sin, mask)
        x = self.layer9(x, cos, sin, mask)
        x = self.layer10(x, cos, sin, mask)
        x = self.layer11(x, cos, sin, mask)
        x, slots = self.sfm11(x, slots)
        pred11_logits = self.sfm11_pred(slots)
        slot11_ids = self.sfm11_tok(slots)
        total_slot_loss = total_slot_loss + self.ce_loss(
            pred11_logits.reshape((-1, SFM_SLOT_VOCAB_SIZE)),
            slot11_ids.reshape((-1,)))

        x = self.layer12(x, cos, sin, mask)
        x = self.layer13(x, cos, sin, mask)
        x = self.layer14(x, cos, sin, mask)
        x = self.layer15(x, cos, sin, mask)
        x = self.layer16(x, cos, sin, mask)
        x = self.layer17(x, cos, sin, mask)
        x, slots = self.sfm17(x, slots)
        pred17_logits = self.sfm17_pred(slots)
        slot17_ids = self.sfm17_tok(slots)
        total_slot_loss = total_slot_loss + self.ce_loss(
            pred17_logits.reshape((-1, SFM_SLOT_VOCAB_SIZE)),
            slot17_ids.reshape((-1,)))

        x = self.layer18(x, cos, sin, mask)
        x = self.layer19(x, cos, sin, mask)
        x = self.layer20(x, cos, sin, mask)
        x = self.layer21(x, cos, sin, mask)
        x = self.layer22(x, cos, sin, mask)
        x = self.layer23(x, cos, sin, mask)
        x, slots = self.sfm23(x, slots)
        pred23_logits = self.sfm23_pred(slots)
        slot23_ids = self.sfm23_tok(slots)
        total_slot_loss = total_slot_loss + self.ce_loss(
            pred23_logits.reshape((-1, SFM_SLOT_VOCAB_SIZE)),
            slot23_ids.reshape((-1,)))

        x = self.norm(x)
        h2 = x.reshape((-1, HIDDEN_DIM))
        logits = self.matmul(h2, self.lm_head)
        logits = logits.reshape(B, x.shape[1], VOCAB_SIZE)
        return logits, total_slot_loss


# ── Training cells ──────────────────────────────────────────────────


class ForwardLossCell(nn.Cell):
    """LM cross-entropy + slot prediction CE loss."""

    def __init__(self, model: Thinker15BModel, cos: Tensor, sin: Tensor,
                 mask: Tensor):
        super().__init__()
        self.model = model
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos = cos
        self.sin = sin
        self.mask = mask

    def construct(self, input_ids: Tensor) -> Tensor:
        logits, slot_loss = self.model(input_ids, self.cos, self.sin,
                                       self.mask)
        logits_t = logits[:, :-1, :]
        labels = input_ids[:, 1:].reshape((-1,))
        ce = self.ce_loss(logits_t.reshape((-1, VOCAB_SIZE)), labels)
        return ce + SLOT_PRED_WEIGHT * slot_loss


class TrainStep(nn.Cell):
    """Manual train step with manual clip-by-global-norm."""

    def __init__(self, forward_loss: ForwardLossCell, optimizer,
                 max_grad_norm: float = MAX_GRAD_NORM):
        super().__init__()
        self.forward_loss = forward_loss
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = Tensor([1.0], ms.float32)
        self.depend = ops.Depend()
        self.max_norm = Tensor(max_grad_norm, ms.float32)
        self.eps = Tensor(1e-6, ms.float32)
        self.one = Tensor(1.0, ms.float32)
        self.sqrt = ops.Sqrt()
        self.minimum = ops.Minimum()

    @ms.jit
    def construct(self, input_ids: Tensor) -> Tensor:
        loss = self.forward_loss(input_ids)
        grads = self.grad_op(self.forward_loss, self.weights)(
            input_ids, self.sens)
        total_norm_sq = Tensor(0.0, ms.float32)
        for g in grads:
            total_norm_sq = total_norm_sq + ops.reduce_sum(g * g)
        global_norm = self.sqrt(total_norm_sq + self.eps)
        clip_coef = self.minimum(self.max_norm / global_norm, self.one)
        clipped = tuple(g * clip_coef for g in grads)
        status = self.optimizer(clipped)
        return self.depend(loss, status)


# ── Helpers ─────────────────────────────────────────────────────────


def count_params(model: nn.Cell) -> tuple:
    total = sum(p.size for p in model.get_parameters())
    return total, total


def _find_data_files(base: str) -> list:
    found = []
    for ext in ("*.jsonl", "*.json", "*.parquet", "*.csv", "*.txt", "*.npy"):
        found.extend(
            glob.glob(os.path.join(base, "**", ext), recursive=True))
    return sorted(found)


def _extract_text(row: dict) -> str:
    for key in ("messages", "conversations"):
        if key in row:
            parts = row[key]
            if isinstance(parts, list):
                return "\n".join(
                    m.get("content", m.get("text", ""))
                    for m in parts if isinstance(m, dict))
    for pair in [("output", "input"), ("response", "instruction"),
                 ("solution", "problem"), ("answer", "question"),
                 ("content", "text")]:
        a, b = pair
        if a in row and b in row:
            return str(row[b]) + "\n" + str(row[a])
        if a in row:
            return str(row[a])
    return str(row)


def load_dataset(dataset_path: str, seq_len: int) -> tuple:
    """Load data from .npy chunks or JSONL files.

    Returns (chunks_array, grpo_list) where:
      - chunks_array: (N, seq_len) int32 or None
      - grpo_list: list of {"text", "reward"} dicts (v2: unused)
    """
    files = _find_data_files(dataset_path)
    log(f"  Found {len(files)} data file(s) in {dataset_path}")

    chunks = None
    grpo_samples = []

    npy_files = [f for f in files if f.endswith(".npy")]
    for nf in npy_files:
        try:
            data = np.load(nf)
            if data.shape[-1] == seq_len:
                chunks = data.astype(np.int32)
                log(f"  Loaded pre-tokenized chunks: {chunks.shape}")
                break
            elif data.shape[-1] != seq_len:
                log(f"  Skipping {nf}: wrong seq_len "
                    f"({data.shape[-1]} vs {seq_len})")
        except Exception as e:
            log(f"  Failed to load {nf}: {e}")

    if chunks is None:
        texts = []
        for fp in files:
            if fp.endswith(".npy"):
                continue
            log(f"  Reading {os.path.basename(fp)} ...")
            try:
                if fp.endswith(".jsonl"):
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                texts.append(_extract_text(data))
                elif fp.endswith(".json"):
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for row in data:
                                texts.append(_extract_text(row))
            except Exception as e:
                log(f"    Error reading {fp}: {e}")

        if texts:
            all_ids = []
            for text in texts:
                for ch in text:
                    all_ids.append(min(ord(ch) % VOCAB_SIZE, VOCAB_SIZE - 1))
                all_ids.append(VOCAB_SIZE - 1)
            total = len(all_ids)
            usable = (total // seq_len) * seq_len
            if usable >= seq_len:
                chunks = np.array(all_ids[:usable], dtype=np.int32
                                  ).reshape(-1, seq_len)
                log(f"  Byte-encoded: {chunks.shape}")

    return chunks, grpo_samples


# ── Main training ───────────────────────────────────────────────────


def main() -> None:
    rank_id = int(os.environ.get("RANK_ID", 0))
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    device_id = int(os.environ.get("DEVICE_ID", rank_id))

    setup_logging(rank_id)
    log(f"Thinker-1.5B v2 training | rank={rank_id}/{rank_size} "
        f"device={device_id}")

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
        device_id=device_id,
        memory_optimize_level="O1",
    )

    use_dp = rank_size > 1
    if use_dp:
        try:
            time.sleep(rank_id * 0.5)
            ms.communication.init()
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=rank_size,
            )
            log(f"DATA_PARALLEL init OK ({rank_size} NPUs)")
        except Exception as e:
            log(f"DATA_PARALLEL init FAILED: {e}")
            log("Falling back to single-device")
            use_dp = False
            rank_size = 1

    log("Building Thinker-1.5B v2 model ...")
    model = Thinker15BModel()

    for i in range(NUM_LAYERS):
        getattr(model, f"layer{i}").recompute()

    total_p, _ = count_params(model)
    log(f"Total params: {total_p:,}")

    log("Loading dataset ...")
    data_chunks, grpo_samples = load_dataset(DATASET_PATH, MAX_SEQ_LEN)
    use_real_data = data_chunks is not None

    if use_real_data:
        log(f"Dataset: {data_chunks.shape[0]} chunks, "
            f"{data_chunks.shape[1]} tok/chunk")
    else:
        log("WARNING: no dataset — using random tokens")

    # Precompute RoPE + mask
    S = MAX_SEQ_LEN
    inv_freq = 1.0 / (ROPE_THETA ** (
        np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    t = np.arange(S, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos_np = np.cos(emb).astype(np.float16)
    sin_np = np.sin(emb).astype(np.float16)
    cos_t = Tensor(cos_np[np.newaxis, np.newaxis, :, :])
    sin_t = Tensor(sin_np[np.newaxis, np.newaxis, :, :])
    causal_np = np.triu(
        np.full((S, S), -1e4, dtype=np.float16), k=1)
    causal_mask = Tensor(causal_np.reshape(1, 1, S, S))

    # LR schedule
    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    if use_real_data:
        max_steps = data_chunks.shape[0] * 5 // rank_size + 100
    else:
        max_steps = 500_000_000 // tokens_per_step + 100

    lr_schedule = []
    for s in range(max_steps):
        if s < WARMUP_STEPS:
            lr = LEARNING_RATE * s / max(1, WARMUP_STEPS)
        else:
            progress = (s - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
            lr = MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (
                1 + math.cos(math.pi * progress))
        lr_schedule.append(lr)
    log(f"LR schedule: {max_steps} steps, warmup={WARMUP_STEPS}")

    optimizer = nn.AdamWeightDecay(
        list(model.trainable_params()),
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        beta1=0.9,
        beta2=0.95,
    )

    forward_loss = ForwardLossCell(model, cos_t, sin_t, causal_mask)
    train_step = None
    actual_bs = BATCH_SIZE

    for bs in [4, 2, 1]:
        try:
            log(f"Building training graph for B={bs} ...")
            ts = TrainStep(forward_loss, optimizer, MAX_GRAD_NORM)
            ts.set_train()
            dummy = Tensor(
                np.random.randint(0, VOCAB_SIZE, (bs, MAX_SEQ_LEN)).astype(
                    np.int32))
            log("  Compiling (first step) ...")
            _ = ts(dummy)
            log(f"  B={bs} compilation OK")
            train_step = ts
            actual_bs = bs
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "alloc" in msg:
                log(f"  B={bs} OOM, trying smaller batch ...")
                del ts
                gc.collect()
                continue
            log(f"  B={bs} FAILED: {e}")
            raise

    if train_step is None:
        log("FATAL: all batch sizes OOM")
        return

    log(f"Training: B={actual_bs}, S={MAX_SEQ_LEN}, "
        f"{actual_bs * MAX_SEQ_LEN:,} tok/step/dev")

    step = 0
    total_tokens = 0
    best_loss = float("inf")
    t_start = time.time()
    tokens_per_step_actual = actual_bs * MAX_SEQ_LEN
    stop_reason = "not started"
    loss_history = []
    best_rolling_avg = float("inf")
    steps_without_improvement = 0
    epochs_completed = 0
    samples_seen = 0
    total_samples = data_chunks.shape[0] if use_real_data else float("inf")
    prev_best_rolling = float("inf")

    log("")
    log("=" * 60)
    log("TRAINING START (v2)")
    log("=" * 60)

    while step < max_steps:
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            stop_reason = f"hard time limit ({elapsed:.0f}s)"
            log(f"Time limit reached — {stop_reason}")
            break

        step_frac = step / max(1, max_steps)

        try:
            if use_real_data:
                indices = [
                    (step * actual_bs + rank_id * actual_bs + j)
                    % data_chunks.shape[0]
                    for j in range(actual_bs)]
                batch_np = data_chunks[indices]
            else:
                batch_np = np.random.randint(
                    0, VOCAB_SIZE,
                    (actual_bs, MAX_SEQ_LEN)).astype(np.int32)
            batch_tensor = Tensor(batch_np)
            loss_val = train_step(batch_tensor)

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "alloc" in msg:
                stop_reason = f"OOM at step {step}"
                log(f"OOM at step {step}! Stopping training.")
                break
            raise

        loss_float = float(loss_val.asnumpy())
        total_tokens += tokens_per_step_actual * rank_size
        samples_seen += actual_bs
        if loss_float < best_loss:
            best_loss = loss_float

        step += 1

        if use_real_data and samples_seen >= total_samples:
            epochs_completed += 1
            samples_seen -= total_samples
            log(f"=== Epoch {epochs_completed} completed at step {step} ===")

        loss_history.append(loss_float)
        if (len(loss_history) >= CONVERGENCE_WINDOW
                and step > WARMUP_STEPS
                and epochs_completed >= MIN_EPOCHS):
            rolling_avg = sum(loss_history[-CONVERGENCE_WINDOW:]
                              ) / CONVERGENCE_WINDOW
            if rolling_avg < best_rolling_avg * (
                    1 - CONVERGENCE_THRESHOLD):
                best_rolling_avg = rolling_avg
                steps_without_improvement = 0
                if rank_id == 0 and rolling_avg < prev_best_rolling:
                    ms.save_checkpoint(
                        model, os.path.join(CKPT_DIR, "best.ckpt"))
                    log(f"New best model saved "
                        f"(rolling_avg={rolling_avg:.4f})")
                    prev_best_rolling = rolling_avg
            else:
                steps_without_improvement += 1

            if (epochs_completed >= MIN_EPOCHS
                    and steps_without_improvement >= CONVERGENCE_PATIENCE):
                stop_reason = (f"converged (rolling_avg={rolling_avg:.4f}, "
                               f"no improvement for "
                               f"{steps_without_improvement} steps)")
                log(f"STOPPING: {stop_reason}")
                break

        if step % 50 == 0 or step <= 3:
            dt = time.time() - t_start
            tps_dev = (tokens_per_step_actual / dt * step
                       if dt > 0 else 0)
            tps_total = tps_dev * rank_size
            log(f"Step {step:>5d} | loss={loss_float:.4f} | "
                f"best={best_loss:.4f} | "
                f"tok/s/dev={tps_dev:.0f} | "
                f"tok/s/total={tps_total:.0f} | "
                f"lr={lr_schedule[min(step, len(lr_schedule)-1)]:.2e} | "
                f"epoch={epochs_completed} | "
                f"elapsed={dt:.0f}s | "
                f"tokens={total_tokens:,}")

        if rank_id == 0 and step % 1000 == 0 and step > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"step_{step}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            log(f"Checkpoint saved: {ckpt_path}")

    elapsed_total = time.time() - t_start
    tps_avg = total_tokens / elapsed_total if elapsed_total > 0 else 0

    if rank_id == 0:
        ms.save_checkpoint(model, os.path.join(CKPT_DIR, "final.ckpt"))
        log("Final checkpoint saved")

    results = {
        "model_name": "Thinker-1.5B v2 (from-scratch + SFM)",
        "version": "v2",
        "total_params": total_p,
        "trainable_params": total_p,
        "batch_size": actual_bs,
        "seq_len": MAX_SEQ_LEN,
        "num_devices": rank_size,
        "total_steps": step,
        "total_tokens": total_tokens,
        "total_time_s": round(elapsed_total, 1),
        "tokens_per_sec_total": round(tps_avg, 1),
        "final_loss": round(loss_float, 4),
        "best_loss": round(best_loss, 4),
        "best_rolling_avg": round(best_rolling_avg, 4)
        if best_rolling_avg < float("inf") else None,
        "epochs_completed": epochs_completed,
        "stop_reason": stop_reason,
        "data_parallel": use_dp,
        "used_real_data": use_real_data,
        "sfm_features": [
            "real_gru", "slot_persistence", "pre_norm",
            "slot_vocab_ce", "learned_initial_slots",
        ],
    }

    results_path = os.path.join(OUTPUT_PATH, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results written to {results_path}")

    log("")
    log("=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    for k, v in results.items():
        log(f"  {k}: {v}")

    if HAS_C2NET:
        try:
            upload_output()
            log("upload_output() completed")
        except Exception as e:
            log(f"upload_output() failed: {e}")

    if _log_fh is not None:
        _log_fh.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        msg = f"FATAL ERROR in main(): {e}\n{traceback.format_exc()}"
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
        for _err_dir in ["/cache/output", OUTPUT_PATH]:
            try:
                os.makedirs(_err_dir, exist_ok=True)
                with open(os.path.join(_err_dir, "error.log"), "w") as f:
                    f.write(msg)
            except Exception:
                pass
        try:
            if HAS_C2NET:
                from c2net.context import upload_output
                upload_output()
        except Exception:
            pass
        raise
