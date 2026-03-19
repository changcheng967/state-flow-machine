"""train.py — Thinker-1.5B from-scratch training with SFM Slot Banks.

~1.35B param decoder-only LM with GQA, trained from scratch on 4x Ascend 910.
SFM Slot Banks at layers 6, 12, 18, 24 with GRU-gated state persistence.

Architecture:
  - 24 transformer layers: hidden=2048, 16Q/4KV heads (GQA), intermediate=5504
  - RoPE (theta=10000), RMSNorm, SwiGLU FFN
  - Tied embedding weights (no separate lm_head)
  - SFM: 16 slots x 256d at layers 5,11,17,23 (0-indexed)
  - Cross-attention: Q from slots, K/V from sequence
  - GRU gated write-back with gate init=-10 (identity at start)

Data:
  - Phase 1 (40%): OpenThoughts-114k from DATASET_PATH (BPE tokenized)
  - Phase 2 (40%): Synthetic execution traces (exec() with restricted builtins)
  - Phase 3 (20%): Synthetic thinkcode agent trajectories (template-based)
  - Curriculum: start 70/30 P1+P2, then 40/40/20 P1+P2+P3

Infrastructure (reused from reference train.py):
  - c2net path discovery + local fallback
  - 4-worker process forking (subprocess.Popen + RANK_ID)
  - Per-worker logging
  - DATA_PARALLEL via HCCL (fallback to single-device)
  - Batch-size OOM fallback (4->2->1)
  - Convergence-based stopping (rolling 200-step, patience=1000)
  - Checkpointing every 1000 steps + best model
  - results.json + upload_output()

Self-contained: MindSpore 2.2 + NumPy + stdlib only. No pip installs.
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
        f"[BOOT {_boot_ts}] pid={_boot_pid} rank={_boot_rank} "
        f"thinker15b_started\n")
    sys.stderr.flush()
    os.makedirs("/cache/output", exist_ok=True)
    with open("/cache/output/boot.log", "a") as _bf:
        _bf.write(f"[{_boot_ts}] pid={_boot_pid} rank={_boot_rank} "
                  f"thinker15b_started\n")
except Exception:
    pass

import subprocess
warnings.filterwarnings("ignore")

# ── Environment vars (BEFORE importing MindSpore) ────────────────────
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
              f"dataset={DATASET_PATH}, pretrain={PRETRAIN_MODEL_PATH}, "
              f"output={OUTPUT_PATH}", flush=True)
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

# ── Model hyperparameters ────────────────────────────────────────────
VOCAB_SIZE = 32000
HIDDEN_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 4
NUM_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 4
HEAD_DIM = 128
INTERMEDIATE_DIM = 5504
NUM_LAYERS = 24
MAX_SEQ_LEN = 2048
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
TIE_WORD_EMBEDDINGS = True

# SFM Slot Banks
SFM_NUM_SLOTS = 16
SFM_SLOT_DIM = 256
SFM_NUM_HEADS = 4
SFM_LAYERS = (5, 11, 17, 23)  # 0-indexed = layers 6,12,18,24

# ── Training hyperparameters ─────────────────────────────────────────
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

RANK_SIZE = 4

# ── Logging ──────────────────────────────────────────────────────────
_log_fh = None


def log(msg: str, level: str = "INFO") -> None:
    """Write to stdout AND the per-worker log file."""
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
    if rank_id == 0:
        path = os.path.join(OUTPUT_PATH, "train.log")
    else:
        path = os.path.join(OUTPUT_PATH, f"worker_{rank_id}.log")
    _log_fh = open(path, "a")
    log(f"Logging to {path}")


def is_main_process() -> bool:
    return os.environ.get("RANK_ID") is None and "--worker" not in sys.argv


def launch_distributed() -> None:
    """Parent: fork 4 child processes, wait, collect results."""
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

# ── MindSpore imports (AFTER env vars) ───────────────────────────────
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.tensor import Tensor


# ══════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS (inline for PYNATIVE_MODE + @ms.jit safety)
# ══════════════════════════════════════════════════════════════════════


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
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.weight = ms.Parameter(
            Tensor(np.ones(dim, dtype=np.float32)), name="norm_weight")
        self.eps = Tensor(eps, ms.float32)

    def construct(self, x: Tensor) -> Tensor:
        x_f = x.astype(ms.float32)
        variance = ops.mean(x_f * x_f, axis=-1, keep_dims=True)
        x_norm = x_f * ops.rsqrt(variance + self.eps)
        return (x_norm * self.weight).astype(x.dtype)


class TransformerBlock(nn.Cell):
    """Single transformer layer: GQA MHA + SwiGLU FFN + RMSNorm."""

    def __init__(self) -> None:
        super().__init__()
        H = HIDDEN_DIM
        NH = NUM_HEADS
        NKV = NUM_KV_HEADS
        HD = HEAD_DIM
        A = INTERMEDIATE_DIM

        self.q_proj = _fp16_dense(H, NH * HD)
        self.k_proj = _fp16_dense(H, NKV * HD)
        self.v_proj = _fp16_dense(H, NKV * HD)
        self.o_proj = _fp16_dense(NH * HD, H)
        self.input_norm = RMSNorm(H)

        self.gate_proj = _fp16_dense(H, A)
        self.up_proj = _fp16_dense(H, A)
        self.down_proj = _fp16_dense(A, H)
        self.ffn_norm = RMSNorm(H)

        self.scale = Tensor(HD ** -0.5, ms.float16)
        self.tile = ops.Tile()
        self.softmax = ops.Softmax(axis=-1)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> Tensor:
        B, S, _ = x.shape
        NH = NUM_HEADS
        NKV = NUM_KV_HEADS
        HD = HEAD_DIM
        NG = NUM_GROUPS

        h = self.input_norm(x)
        Q = self.q_proj(h).reshape(B, S, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(h).reshape(B, S, NKV, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(h).reshape(B, S, NKV, HD).transpose(0, 2, 1, 3)

        # GQA: tile KV heads to match Q heads
        K = self.tile(K, (1, NG, 1, 1))
        V = self.tile(V, (1, NG, 1, 1))

        # RoPE (inline)
        HD2 = HD // 2
        Q = Q * cos + ops.concat([-Q[..., HD2:], Q[..., :HD2]], -1) * sin
        K = K * cos + ops.concat([-K[..., HD2:], K[..., :HD2]], -1) * sin

        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = attn + mask[:, :, :S, :S]
        attn = self.softmax(attn)
        out = ops.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, NH * HD)
        out = self.o_proj(out)
        x = x + out

        h = self.ffn_norm(x)
        gate = ops.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


class SFMSlotBank(nn.Cell):
    """State-Flow Machine Slot Bank with GRU-gated state persistence.

    Cross-attention: Q from slots, K/V from sequence (Q queries slots,
    attending to the sequence to decide what to read).
    GRU update: gated combination of old slots and new information.
    Gated write-back: sigmoid(gate) * output_proj(mean(slot_out)) added
    to residual. Gate initialised to -10 so sigmoid≈0 at start (identity).
    """

    def __init__(self) -> None:
        super().__init__()
        H = HIDDEN_DIM
        NS = SFM_NUM_SLOTS
        SD = SFM_SLOT_DIM
        NH = SFM_NUM_HEADS
        self.head_dim = SD // NH

        # Cross-attention: Q from slots, K/V from sequence
        self.q_proj = _fp16_dense(SD, NH * self.head_dim)
        self.k_proj = _fp16_dense(H, NH * self.head_dim)
        self.v_proj = _fp16_dense(H, NH * self.head_dim)
        self.out_proj = _fp16_dense(NH * self.head_dim, H)

        # GRU gates for slot update
        gru_in = SD + NH * self.head_dim  # slots + slot_attn_output
        self.W_z = _fp16_dense(gru_in, SD, has_bias=True)
        self.W_r = _fp16_dense(gru_in, SD, has_bias=True)
        self.W_h = _fp16_dense(gru_in, SD, has_bias=True)

        # Gated write-back to residual stream
        self.write_proj = _fp16_dense(SD, H)
        self.gate = ms.Parameter(
            Tensor(np.array([-10.0], dtype=np.float32)), name="sfm_gate")

        self.scale = Tensor(self.head_dim ** -0.5, ms.float16)
        self.softmax = ops.Softmax(axis=-1)
        self.sigmoid = ops.Sigmoid()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.tile = ops.Tile()

    def construct(self, hidden: Tensor, slots: Tensor) -> tuple:
        """Process hidden through slot bank.

        Args:
            hidden: (B, S, H) from transformer layer
            slots: (B, NS, SD) current slot state

        Returns:
            (modified_hidden, new_slots)
        """
        B, S, _ = hidden.shape
        NS = SFM_NUM_SLOTS
        NH = SFM_NUM_HEADS
        HD = self.head_dim

        # Cross-attention: Q from slots, K/V from sequence
        Q = self.q_proj(slots).reshape(B, NS, NH, HD).transpose(0, 2, 1, 3)
        K = self.k_proj(hidden).reshape(B, S, NH, HD).transpose(0, 2, 1, 3)
        V = self.v_proj(hidden).reshape(B, S, NH, HD).transpose(0, 2, 1, 3)

        # (B, NH, NS, HD) @ (B, NH, HD, S) -> (B, NH, NS, S)
        attn = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        # (B, NH, NS, S) @ (B, NH, S, HD) -> (B, NH, NS, HD)
        slot_out = ops.matmul(attn, V)
        slot_out = slot_out.transpose(0, 2, 1, 3).reshape(B, NS, -1)

        # GRU update: concat([old_slots, slot_read]) -> new_slots
        gru_input = ops.concat([slots, slot_out], axis=-1)
        z = self.sigmoid(self.W_z(gru_input))
        r = self.sigmoid(self.W_r(gru_input))
        h_cand = ops.tanh(self.W_h(ops.concat([r * slots, slot_out], -1)))
        new_slots = (1.0 - z) * slots + z * h_cand

        # Gated write-back: mean(slot_out) -> residual
        slot_mean = self.mean(new_slots, 1)  # (B, 1, SD)
        writeback = self.write_proj(slot_mean)  # (B, 1, H)
        writeback = self.tile(writeback, (1, 1, S))  # (B, S, H)
        modified = hidden + self.sigmoid(self.gate.astype(hidden.dtype)) * \
            writeback.astype(hidden.dtype)

        return modified, new_slots


class Thinker15BModel(nn.Cell):
    """~1.35B decoder-only LM with GQA + SFM Slot Banks.

    Construct is UNROLLED for @ms.jit safety. Slot persistence across
    all 4 SFM layers (slots thread through the forward pass).
    Tied embedding weights: logits = hidden @ embedding_table.T
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.norm = RMSNorm(HIDDEN_DIM)
        self.matmul = ops.MatMul(transpose_b=True)

        # Initial slots — learned parameter (broadcast to batch)
        self.initial_slots = ms.Parameter(
            Tensor(np.random.randn(1, SFM_NUM_SLOTS, SFM_SLOT_DIM).astype(
                np.float16) * 0.02),
            name="initial_slots")

        # 24 transformer layers (unrolled)
        for i in range(NUM_LAYERS):
            setattr(self, f"layer{i}", TransformerBlock())

        # 4 SFM banks at layers 5, 11, 17, 23
        for idx, layer_idx in enumerate(SFM_LAYERS):
            setattr(self, f"sfm{idx}", SFMSlotBank())

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> Tensor:
        """Forward pass. Returns logits (B, S, VOCAB_SIZE)."""
        B = input_ids.shape[0]
        x = self.embedding(input_ids).astype(ms.float16)

        # Initialize slots
        slots = self.tile(self.initial_slots,
                          (B, 1, 1))  # (B, NS, SD)

        sfm_idx = 0
        for i in range(NUM_LAYERS):
            x = getattr(self, f"layer{i}")(x, cos, sin, mask)
            if i in SFM_LAYERS:
                x, slots = getattr(self, f"sfm{sfm_idx}")(x, slots)
                sfm_idx += 1

        x = self.norm(x)
        h2 = x.reshape((-1, HIDDEN_DIM))
        if TIE_WORD_EMBEDDINGS:
            logits = self.matmul(h2, self.embedding.embedding_table)
        else:
            logits = self.matmul(h2, self.lm_head)
        return logits.reshape(B, x.shape[1], VOCAB_SIZE)


# ── Training cells ───────────────────────────────────────────────────


class ForwardLossCell(nn.Cell):
    """Language model cross-entropy loss."""

    def __init__(self, model: Thinker15BModel, cos: Tensor, sin: Tensor,
                 mask: Tensor):
        super().__init__()
        self.model = model
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos = cos
        self.sin = sin
        self.mask = mask

    def construct(self, input_ids: Tensor) -> Tensor:
        logits = self.model(input_ids, self.cos, self.sin, self.mask)
        logits_t = logits[:, :-1, :]
        labels = input_ids[:, 1:].reshape((-1,))
        return self.ce_loss(logits_t.reshape((-1, VOCAB_SIZE)), labels)


class TrainStep(nn.Cell):
    """Manual train step with manual clip-by-global-norm.

    Uses @ms.jit for compilation. Manual gradient clipping avoids
    nn.ClipByGlobalNorm issues in MS 2.2.
    """

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


# ── Helpers ──────────────────────────────────────────────────────────


def count_params(model: nn.Cell) -> tuple:
    total = sum(p.size for p in model.get_parameters())
    trainable = sum(p.size for p in model.trainable_params())
    return total, trainable


def _find_data_files(base: str) -> list:
    """Recursively find data files under base."""
    found = []
    for ext in ("*.jsonl", "*.json", "*.parquet", "*.csv", "*.txt",
                "*.npy", "*.arrow"):
        found.extend(glob.glob(os.path.join(base, "**", ext),
                                recursive=True))
    return sorted(found)


def _extract_text(row: dict) -> str:
    """Extract text content from a dataset row (various schema formats)."""
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


# ── BPE Tokenizer ────────────────────────────────────────────────────


def _bytes_to_unicode():
    """GPT-2/Qwen2.5 byte-to-unicode mapping."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_UNICODE = None


def _get_byte_unicode():
    global _BYTE_UNICODE
    if _BYTE_UNICODE is None:
        _BYTE_UNICODE = _bytes_to_unicode()
    return _BYTE_UNICODE


def _get_byte_token(byte_val, vocab):
    """Get vocab token for a single byte value."""
    b2u = _get_byte_unicode()
    ch = b2u[byte_val]
    if ch in vocab:
        return vocab[ch]
    hex_tok = f"<0x{byte_val:02X}>"
    if hex_tok in vocab:
        return vocab[hex_tok]
    hex_tok = f"<0x{byte_val:02x}>"
    if hex_tok in vocab:
        return vocab[hex_tok]
    return 0


def _try_load_hf_tokenizer(pretrain_path: str):
    """Load BPE tokenizer from HuggingFace tokenizer.json.

    Returns (vocab, merge_priority, eos_id) or None on failure.
    """
    tok_path = None
    for cand in [os.path.join(pretrain_path, "tokenizer.json")]:
        if os.path.isfile(cand):
            tok_path = cand
            break
    if tok_path is None:
        results = glob.glob(os.path.join(pretrain_path, "**",
                                          "tokenizer.json"), recursive=True)
        if results:
            tok_path = results[0]
    if tok_path is None:
        return None
    try:
        with open(tok_path) as f:
            tok_data = json.load(f)
        vocab_dict = tok_data.get("model", {}).get("vocab", {})
        merges_raw = tok_data.get("model", {}).get("merges", [])
        if not vocab_dict:
            return None

        token_to_id = {}
        for token, idx in vocab_dict.items():
            token_to_id[token] = int(idx)

        added = tok_data.get("added_tokens", [])
        eos_id = 1  # Default
        for entry in added:
            if isinstance(entry, dict):
                if entry.get("content") == "<|endoftext|>":
                    eos_id = entry.get("id", eos_id)
                elif entry.get("content") == "":
                    eos_id = entry.get("id", eos_id)

        merge_priority = {}
        for idx, merge_str in enumerate(merges_raw):
            parts = merge_str.split(" ", 1)
            if len(parts) == 2:
                merge_priority[(parts[0], parts[1])] = idx

        log(f"  Loaded HF tokenizer: {len(token_to_id)} vocab, "
            f"{len(merges_raw)} merges from {tok_path}")
        return token_to_id, merge_priority, eos_id
    except Exception as e:
        log(f"  Failed to parse tokenizer.json: {e}")
        return None


def train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE, min_count=2):
    """Train a byte-level BPE tokenizer on texts.

    Returns (vocab, merge_priority, eos_id).
    """
    log(f"  Training BPE tokenizer on {len(texts)} texts "
        f"(target vocab={vocab_size})...")
    b2u = _get_byte_unicode()

    # Pre-tokenize: split on whitespace boundaries
    word_freqs = {}
    for text in texts:
        words = []
        current = []
        for ch in text:
            if ch in (" ", "\t", "\n", "\r"):
                if current:
                    words.append("".join(current))
                    current = []
                if ch == " ":
                    words.append(None)
                elif ch == "\n":
                    words.append("\n")
            else:
                current.append(ch)
        if current:
            words.append("".join(current))

        merged_words = []
        i = 0
        while i < len(words):
            if words[i] is None and i + 1 < len(words) and \
                    words[i + 1] is not None:
                merged_words.append("\u0120" + words[i + 1])
                i += 2
            elif words[i] is None:
                merged_words.append("\u0120")
                i += 1
            else:
                merged_words.append(words[i])
                i += 1

        for word in merged_words:
            word_bytes = word.encode("utf-8", errors="replace")
            tokens = tuple(b2u[b] for b in word_bytes)
            word_freqs[tokens] = word_freqs.get(tokens, 0) + 1

    # Base vocab: all 256 byte tokens + special tokens
    vocab = {}
    for byte_val in range(256):
        ch = b2u[byte_val]
        if ch not in vocab:
            vocab[ch] = len(vocab)
    special_tokens = ["<pad>", "<eos>", "<unk>"]
    for tok in special_tokens:
        vocab[tok] = len(vocab)
    eos_id = vocab["<eos>"]

    # Merge pairs iteratively
    merge_priority = {}
    for merge_idx in range(vocab_size - len(vocab)):
        # Count pairs
        pair_counts = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_pair] < min_count:
            break

        new_tok = best_pair[0] + best_pair[1]
        vocab[new_tok] = len(vocab)
        merge_priority[best_pair] = merge_idx

        # Apply merge to all words
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and \
                        word[i + 1] == best_pair[1]:
                    new_word.append(new_tok)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = new_word_freqs.get(
                tuple(new_word), 0) + freq
        word_freqs = new_word_freqs

        if (merge_idx + 1) % 1000 == 0:
            log(f"    BPE merge {merge_idx + 1}/{vocab_size - len(vocab)}, "
                f"pair={best_pair}, count={pair_counts[best_pair]}, "
                f"vocab={len(vocab)}")

    log(f"  BPE training done: {len(vocab)} tokens, "
        f"{len(merge_priority)} merges")
    return vocab, merge_priority, eos_id


def tokenize_text(text, vocab, merge_priority):
    """Tokenize a single text using byte-level BPE."""
    b2u = _get_byte_unicode()

    # Pre-tokenize
    words = []
    current = []
    for ch in text:
        if ch in (" ", "\t", "\n", "\r"):
            if current:
                words.append("".join(current))
                current = []
            if ch == " ":
                words.append(None)
            elif ch == "\n":
                words.append("\n")
        else:
            current.append(ch)
    if current:
        words.append("".join(current))

    merged_words = []
    i = 0
    while i < len(words):
        if words[i] is None and i + 1 < len(words) and \
                words[i + 1] is not None:
            merged_words.append("\u0120" + words[i + 1])
            i += 2
        elif words[i] is None:
            merged_words.append("\u0120")
            i += 1
        else:
            merged_words.append(words[i])
            i += 1

    token_ids = []
    for word in merged_words:
        word_bytes = word.encode("utf-8", errors="replace")
        tokens = [b2u[b] for b in word_bytes]

        # Apply BPE merges
        while len(tokens) >= 2:
            best_pair = None
            best_pri = float("inf")
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pri = merge_priority.get(pair)
                if pri is not None and pri < best_pri:
                    best_pri = pri
                    best_pair = pair
                    best_idx = j
            if best_pair is None:
                break
            new_tokens = tokens[:best_idx]
            new_tokens.append(best_pair[0] + best_pair[1])
            new_tokens.extend(tokens[best_idx + 2:])
            tokens = new_tokens

        for tok in tokens:
            tid = vocab.get(tok)
            if tid is not None:
                token_ids.append(tid)
            else:
                # Byte fallback
                for ch_tok in tok:
                    byte_val = ord(ch_tok)
                    if byte_val < 256:
                        token_ids.append(_get_byte_token(byte_val, vocab))
                    else:
                        token_ids.append(vocab.get("<unk>", 0))

    return token_ids


def _tokenize_texts(texts, vocab, merge_priority, eos_id, seq_len):
    """Convert text strings to token-id chunks of fixed seq_len.

    Returns (num_chunks, seq_len) int32 array, or None on failure.
    """
    log(f"  Tokenizing {len(texts)} texts ...")
    t0 = time.time()
    all_ids = []
    for i, text in enumerate(texts):
        ids = tokenize_text(text, vocab, merge_priority)
        all_ids.extend(ids)
        all_ids.append(eos_id)
        if (i + 1) % 10000 == 0:
            log(f"    {i + 1}/{len(texts)} done "
                f"({len(all_ids):,} tokens) ...")
    all_ids = np.array(all_ids, dtype=np.int32)
    log(f"  Tokenization done in {time.time() - t0:.1f}s — "
        f"{len(all_ids):,} tokens")

    total = len(all_ids)
    usable = (total // seq_len) * seq_len
    if usable < seq_len:
        log("  WARNING: dataset too small for even one chunk")
        return None
    chunks = all_ids[:usable].reshape(-1, seq_len)
    log(f"  {len(chunks)} chunks of {seq_len} tokens from "
        f"{total:,} tokens")
    return chunks


# ── Synthetic data generation ────────────────────────────────────────

# Restricted builtins for exec()
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range,
    "int": int, "float": float, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "set": set, "bool": bool,
    "sum": sum, "round": round, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "isinstance": isinstance, "print": lambda *a: None,
    "True": True, "False": False, "None": None,
}


def _gen_stage1_variables(n_samples: int) -> list:
    """Stage 1: Variable assignments and arithmetic."""
    texts = []
    var_names = [f"var_{i}" for i in range(20)] + \
               [f"x{i}" for i in range(10)] + \
               ["result", "total", "count", "value", "temp", "ans"]
    ops = ["+", "-", "*", "//", "%"]

    for _ in range(n_samples):
        lines = []
        n_vars = random.randint(3, 10)
        used = random.sample(var_names, min(n_vars, len(var_names)))
        for v in used:
            val = random.randint(-100, 100)
            lines.append(f"{v} = {val}")
        n_expr = random.randint(2, 8)
        for _ in range(n_expr):
            v1 = random.choice(used)
            op = random.choice(ops)
            if op == "//":
                v2 = random.choice(used)
                lines.append(f"{v1} = {v1} {op} max(1, abs({v2}))")
            elif op == "%":
                v2 = random.choice(used)
                lines.append(f"{v1} = {v1} {op} max(1, abs({v2}))")
            else:
                if random.random() < 0.7:
                    v2 = random.choice(used)
                else:
                    v2 = random.randint(-50, 50)
                lines.append(f"{v1} = {v1} {op} {v2}")
        out_var = random.choice(used)
        lines.append(f"print({out_var})")
        texts.append("\n".join(lines))
    return texts


def _gen_stage2_control_flow(n_samples: int) -> list:
    """Stage 2: Control flow (if/elif/else, for, while)."""
    texts = []
    var_names = ["i", "j", "k", "n", "x", "y", "s", "total", "found",
                 "result", "count", "max_val", "min_val", "data"]

    for _ in range(n_samples):
        lines = []
        kind = random.choice(["if", "for", "while", "nested"])

        if kind == "if":
            n = random.randint(5, 15)
            lines.append(f"n = {n}")
            for _ in range(random.randint(2, 5)):
                v = random.choice(var_names)
                val = random.randint(-100, 100)
                lines.append(f"{v} = {val}")
            threshold = random.randint(0, 50)
            lines.append(f"if n > {threshold}:")
            lines.append(f"    result = n * 2")
            lines.append(f"elif n > {threshold // 2}:")
            lines.append(f"    result = n + 10")
            lines.append(f"else:")
            lines.append(f"    result = 0")
            lines.append(f"print(result)")

        elif kind == "for":
            n = random.randint(5, 20)
            lines.append(f"total = 0")
            lines.append(f"for i in range({n}):")
            op = random.choice(["+", "*"])
            val = random.randint(1, 10)
            if op == "+":
                lines.append(f"    total = total + {val}")
            else:
                lines.append(f"    total = total * {val}")
            lines.append(f"print(total)")

        elif kind == "while":
            start = random.randint(1, 50)
            lines.append(f"x = {start}")
            target = random.randint(100, 500)
            step = random.randint(2, 10)
            lines.append(f"while x < {target}:")
            lines.append(f"    x = x + {step}")
            lines.append(f"print(x)")

        elif kind == "nested":
            n = random.randint(3, 10)
            lines.append(f"result = 0")
            lines.append(f"for i in range({n}):")
            lines.append(f"    if i % 2 == 0:")
            lines.append(f"        result = result + i")
            lines.append(f"    else:")
            lines.append(f"        result = result - i // 2")
            lines.append(f"print(result)")

        texts.append("\n".join(lines))
    return texts


def _gen_stage3_functions(n_samples: int) -> list:
    """Stage 3: Functions with lists and basic algorithms."""
    texts = []

    for _ in range(n_samples):
        lines = []
        kind = random.choice(["sort_search", "math_func", "list_comp",
                               "fibonacci"])

        if kind == "sort_search":
            fn = random.choice(["find_max", "find_min", "binary_search",
                                "bubble_sort"])
            if fn == "find_max":
                lines.append("def find_max(arr):")
                lines.append("    m = arr[0]")
                lines.append("    for x in arr:")
                lines.append("        if x > m:")
                lines.append("            m = x")
                lines.append("    return m")
            elif fn == "find_min":
                lines.append("def find_min(arr):")
                lines.append("    m = arr[0]")
                lines.append("    for x in arr:")
                lines.append("        if x < m:")
                lines.append("            m = x")
                lines.append("    return m")
            elif fn == "bubble_sort":
                lines.append("def bubble_sort(arr):")
                lines.append("    n = len(arr)")
                lines.append("    for i in range(n):")
                lines.append("        for j in range(n - i - 1):")
                lines.append("            if arr[j] > arr[j+1]:")
                lines.append("                arr[j], arr[j+1] = "
                             "arr[j+1], arr[j]")
                lines.append("    return arr")
            elif fn == "binary_search":
                lines.append("def binary_search(arr, target):")
                lines.append("    lo, hi = 0, len(arr) - 1")
                lines.append("    while lo <= hi:")
                lines.append("        mid = (lo + hi) // 2")
                lines.append("        if arr[mid] == target:")
                lines.append("            return mid")
                lines.append("        elif arr[mid] < target:")
                lines.append("            lo = mid + 1")
                lines.append("        else:")
                lines.append("            hi = mid - 1")
                lines.append("    return -1")
            # Test
            arr = [random.randint(1, 100) for _ in range(random.randint(5, 15))]
            lines.append(f"data = {arr}")
            lines.append(f"print({fn}(data))")

        elif kind == "math_func":
            fn = random.choice(["factorial", "gcd", "is_prime", "power"])
            if fn == "factorial":
                lines.append("def factorial(n):")
                lines.append("    if n <= 1:")
                lines.append("        return 1")
                lines.append("    return n * factorial(n - 1)")
                lines.append(f"print(factorial({random.randint(3, 12)}))")
            elif fn == "gcd":
                lines.append("def gcd(a, b):")
                lines.append("    while b:")
                lines.append("        a, b = b, a % b")
                lines.append("    return a")
                a, b = random.randint(10, 100), random.randint(10, 100)
                lines.append(f"print(gcd({a}, {b}))")
            elif fn == "is_prime":
                lines.append("def is_prime(n):")
                lines.append("    if n < 2:")
                lines.append("        return False")
                lines.append("    for i in range(2, int(n**0.5) + 1):")
                lines.append("        if n % i == 0:")
                lines.append("            return False")
                lines.append("    return True")
                lines.append(f"print(is_prime({random.randint(2, 100)}))")
            elif fn == "power":
                lines.append("def power(base, exp):")
                lines.append("    result = 1")
                lines.append("    for _ in range(exp):")
                lines.append("        result = result * base")
                lines.append("    return result")
                b, e = random.randint(2, 10), random.randint(2, 8)
                lines.append(f"print(power({b}, {e}))")

        elif kind == "list_comp":
            lines.append(f"data = {[random.randint(1, 50)
                                       for _ in range(random.randint(5, 15))]}")
            op = random.choice(["evens", "squares", "filter_gt"])
            if op == "evens":
                lines.append("evens = [x for x in data if x % 2 == 0]")
                lines.append("print(evens)")
            elif op == "squares":
                lines.append("squares = [x*x for x in data]")
                lines.append("print(squares)")
            elif op == "filter_gt":
                threshold = random.randint(10, 30)
                lines.append(f"big = [x for x in data if x > {threshold}]")
                lines.append("print(big)")

        elif kind == "fibonacci":
            lines.append("def fibonacci(n):")
            lines.append("    if n <= 1:")
            lines.append("        return n")
            lines.append("    a, b = 0, 1")
            lines.append("    for _ in range(n - 1):")
            lines.append("        a, b = b, a + b")
            lines.append("    return b")
            lines.append(f"print(fibonacci({random.randint(5, 20)}))")

        texts.append("\n".join(lines))
    return texts


def _gen_stage4_docs(n_samples: int) -> list:
    """Stage 4: Fictional API documentation with code examples."""
    texts = []
    modules = ["datastore", "pipeline", "scheduler", "config", "auth",
               "cache", "logger", "metrics", "client", "registry"]

    for _ in range(n_samples):
        lines = []
        mod = random.choice(modules)
        cap_mod = mod.capitalize()

        # Module docstring
        lines.append(f"\"\"\"{cap_mod} module - provides {mod} "
                     f"functionality.")
        lines.append(f"\"\"\"")
        lines.append("")

        # Classes
        n_classes = random.randint(1, 3)
        for ci in range(n_classes):
            cls_choices = ['Manager', 'Handler', 'Client', 'Config',
                           'Store', 'Builder']
            cls_name = f"{cap_mod}{random.choice(cls_choices)}"
            lines.append(f"class {cls_name}:")
            lines.append(f"    \"\"\"Manages {mod} operations.\"\"\"")
            lines.append("")

            # __init__
            params = []
            for pi in range(random.randint(1, 4)):
                pname = random.choice(["name", "path", "size", "timeout",
                                       "retries", "verbose", "debug",
                                       "buffer_size", "max_items",
                                       "auto_commit"])
                ptype = random.choice(["str", "int", "bool", "float"])
                pdefault = random.choice(["None", "True", "False", "0",
                                          "100", "''"])
                params.append(f"{pname}: {ptype} = {pdefault}")
            lines.append(f"    def __init__(self, {', '.join(params)}):")
            for p in params:
                pname = p.split(":")[0].strip()
                lines.append(f"        self.{pname} = {pname}")
            lines.append("")

            # Methods
            n_methods = random.randint(2, 5)
            for mi in range(n_methods):
                mname = random.choice(
                    ["get", "set", "create", "delete", "update", "list",
                     "find", "validate", "process", "transform", "connect",
                     "disconnect", "sync", "flush"])
                ret_type = random.choice(["None", "bool", "str", "int",
                                          "list", "dict"])
                lines.append(f"    def {mname}(self, key: str) -> {ret_type}:")
                lines.append(f"        \"\"\"{mname} a {mod} resource.\"\"\"")
                # Body
                if mname in ("get", "find"):
                    lines.append(f"        return self._data.get(key)")
                elif mname in ("create", "set"):
                    lines.append(f"        self._data[key] = value")
                    lines.append(f"        return True")
                elif mname in ("delete", "remove"):
                    lines.append(f"        if key in self._data:")
                    lines.append(f"            del self._data[key]")
                    lines.append(f"            return True")
                    lines.append(f"        return False")
                elif mname == "list":
                    lines.append(f"        return list(self._data.keys())")
                elif mname == "validate":
                    lines.append(f"        return isinstance(key, str) "
                                 f"and len(key) > 0")
                else:
                    lines.append(f"        pass")
                lines.append("")

            # Usage example
            lines.append("# Example usage:")
            args = []
            for p in params:
                pname = p.split(":")[0].strip()
                if "str" in p:
                    args.append(f"'{pname}_value'")
                elif "bool" in p:
                    args.append("True")
                elif "int" in p or "float" in p:
                    args.append(str(random.randint(1, 100)))
                else:
                    args.append("None")
            lines.append(f"mgr = {cls_name}({', '.join(args)})")
            if n_methods > 0:
                mname = random.choice(
                    ["get", "set", "create", "delete", "list", "find"])
                lines.append(f"result = mgr.{mname}('example_key')")
                lines.append(f"print(result)")
            lines.append("")

        texts.append("\n".join(lines))
    return texts


def _gen_stage5_errors(n_samples: int) -> list:
    """Stage 5: Error handling and edge cases."""
    texts = []

    for _ in range(n_samples):
        lines = []
        kind = random.choice(["try_except", "validation", "edge_case"])

        if kind == "try_except":
            lines.append("def safe_divide(a, b):")
            lines.append("    try:")
            lines.append("        return a / b")
            lines.append("    except ZeroDivisionError:")
            lines.append("        return None")
            lines.append("")
            lines.append("def safe_index(arr, idx):")
            lines.append("    try:")
            lines.append("        return arr[idx]")
            lines.append("    except (IndexError, TypeError):")
            lines.append("        return -1")
            lines.append("")
            lines.append(f"print(safe_divide({random.randint(1, 100)}, "
                         f"{random.choice([0, random.randint(1, 10)])}))")

        elif kind == "validation":
            fn = random.choice(["validate_age", "validate_email",
                                "validate_score"])
            if fn == "validate_age":
                lines.append("def validate_age(age):")
                lines.append("    if not isinstance(age, (int, float)):")
                lines.append("        raise ValueError('Age must be a number')")
                lines.append("    if age < 0 or age > 150:")
                lines.append("        raise ValueError('Age out of range')")
                lines.append("    return True")
            elif fn == "validate_score":
                lines.append("def validate_score(score):")
                lines.append("    if not isinstance(score, (int, float)):")
                lines.append("        return False")
                lines.append("    return 0 <= score <= 100")
            else:
                lines.append("def validate_positive(n):")
                lines.append("    return isinstance(n, (int, float)) and n > 0")
            lines.append("")
            lines.append(f"print({fn}({random.randint(1, 100)}))")

        elif kind == "edge_case":
            lines.append("def safe_sum(arr):")
            lines.append("    if not arr:")
            lines.append("        return 0")
            lines.append("    total = 0")
            lines.append("    for x in arr:")
            lines.append("        if isinstance(x, (int, float)):")
            lines.append("            total = total + x")
            lines.append("    return total")
            lines.append("")
            cases = random.choice([
                "[]", "[1, 2, 3]", "[1, 'a', 3, None]",
                "[-1, -2, -3]", "[0]"
            ])
            lines.append(f"print(safe_sum({cases}))")

        texts.append("\n".join(lines))
    return texts


def _gen_stage6_trajectories(n_samples: int) -> list:
    """Stage 6: Synthetic thinkcode agent trajectories (ReAct-style)."""
    texts = []
    tasks = [
        "sort a list of numbers",
        "find the maximum value in a list",
        "calculate the sum of even numbers",
        "reverse a string",
        "count occurrences of an element",
        "check if a number is prime",
        "find duplicates in a list",
        "merge two sorted lists",
        "calculate factorial",
        "find the median of a list",
    ]

    for _ in range(n_samples):
        task = random.choice(tasks)
        lines = []
        lines.append(f"Task: {task}")
        lines.append("")

        # Thought / Action / Observation cycles
        n_steps = random.randint(3, 8)
        for si in range(n_steps):
            lines.append(f"Thought {si+1}: Let me think about step {si+1} "
                         f"of {task}.")

            if si == 0:
                lines.append(f"Action: Understand the input format and "
                             f"requirements.")
                lines.append(f"Observation: The task requires processing "
                             f"the input data step by step.")
            elif si < n_steps - 1:
                action = random.choice([
                    "Break down the problem into smaller sub-problems.",
                    "Consider edge cases and boundary conditions.",
                    "Apply the appropriate algorithm.",
                    "Verify the intermediate result.",
                    "Handle potential errors gracefully.",
                ])
                lines.append(f"Action: {action}")
                lines.append(f"Observation: Intermediate result looks "
                             f"correct. Moving to next step.")
            else:
                lines.append(f"Action: Combine all results and produce "
                             f"the final answer.")

            lines.append("")

        lines.append(f"Answer: The solution handles {task} correctly.")
        lines.append(f"```python")
        # Generate a simple solution
        if "sort" in task:
            lines.append("def solution(arr):")
            lines.append("    return sorted(arr)")
        elif "maximum" in task or "max" in task:
            lines.append("def solution(arr):")
            lines.append("    return max(arr)")
        elif "sum" in task and "even" in task:
            lines.append("def solution(arr):")
            lines.append("    return sum(x for x in arr if x % 2 == 0)")
        elif "reverse" in task:
            lines.append("def solution(s):")
            lines.append("    return s[::-1]")
        elif "prime" in task:
            lines.append("def solution(n):")
            lines.append("    if n < 2: return False")
            lines.append("    for i in range(2, int(n**0.5) + 1):")
            lines.append("        if n % i == 0: return False")
            lines.append("    return True")
        elif "factorial" in task:
            lines.append("def solution(n):")
            lines.append("    r = 1")
            lines.append("    for i in range(2, n+1): r *= i")
            lines.append("    return r")
        else:
            lines.append("def solution(data):")
            lines.append("    return data")
        lines.append("```")
        texts.append("\n".join(lines))
    return texts


def generate_synthetic_data(n_total: int = 50000) -> list:
    """Generate synthetic training data from all stages.

    Stage distribution:
      1. Variables + arithmetic: 20%
      2. Control flow: 20%
      3. Functions + algorithms: 20%
      4. API documentation: 20%
      5. Error handling: 10%
      6. Agent trajectories: 10%
    """
    log(f"Generating {n_total} synthetic texts ...")
    t0 = time.time()
    all_texts = []

    stage_gens = [
        (_gen_stage1_variables, 0.20),
        (_gen_stage2_control_flow, 0.20),
        (_gen_stage3_functions, 0.20),
        (_gen_stage4_docs, 0.20),
        (_gen_stage5_errors, 0.10),
        (_gen_stage6_trajectories, 0.10),
    ]

    for gen_fn, frac in stage_gens:
        n = int(n_total * frac)
        log(f"  Stage: {gen_fn.__name__} ({n} samples) ...")
        texts = gen_fn(n)
        all_texts.extend(texts)

    random.shuffle(all_texts)
    log(f"Synthetic generation done in {time.time() - t0:.1f}s — "
        f"{len(all_texts)} texts")
    return all_texts


def load_dataset(dataset_path: str, seq_len: int) -> tuple:
    """Load and tokenize dataset.

    Strategy:
      1. Try to load OpenThoughts-114k from DATASET_PATH (real data)
      2. Try to load pre-tokenized .npy chunks
      3. Generate synthetic data as fallback
      4. Train BPE tokenizer on available text

    Returns (chunks_array, has_real_data).
    """
    files = _find_data_files(dataset_path)
    log(f"  Found {len(files)} data file(s) in {dataset_path}")

    # Try pre-tokenized .npy
    npy_files = [f for f in files if f.endswith(".npy")]
    for nf in npy_files:
        try:
            data = np.load(nf)
            if data.ndim == 2 and data.shape[-1] == seq_len:
                chunks = data.astype(np.int32)
                log(f"  Loaded pre-tokenized: {chunks.shape}")
                return chunks, True
        except Exception:
            pass

    # Load texts from data files
    texts = []

    # Read real data files
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
                            texts.append(_extract_text(json.loads(line)))
            elif fp.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for row in data:
                            texts.append(_extract_text(row))
                    elif isinstance(data, dict):
                        texts.append(_extract_text(data))
            elif fp.endswith(".txt"):
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif fp.endswith(".parquet"):
                loaded = False
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(fp)
                    cols = table.to_pydict()
                    for i in range(table.num_rows):
                        row = {col: cols[col][i] for col in cols}
                        texts.append(_extract_text(row))
                    loaded = True
                except Exception:
                    pass
                if not loaded:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(fp)
                        for val in df.itertuples(index=False):
                            texts.append(_extract_text(val._asdict()))
                    except Exception:
                        pass
        except Exception as e:
            log(f"    Error reading {fp}: {e}")

    has_real_data = len(texts) > 0
    if has_real_data:
        # Deduplicate
        texts = list(set(t for t in texts if len(t.strip()) > 10))
        log(f"  Loaded {len(texts)} unique real text samples")

    # Generate synthetic data
    target_synth = 50000
    if len(texts) < 1000:
        log(f"  Few real texts ({len(texts)}), generating synthetic data...")
        synth_texts = generate_synthetic_data(target_synth)
        texts.extend(synth_texts)
        log(f"  Total texts: {len(texts)} "
            f"({len(synth_texts)} synthetic)")

    if not texts:
        log("  WARNING: no text data at all")
        return None, False

    # Try to load HF tokenizer first
    tok_result = _try_load_hf_tokenizer(PRETRAIN_MODEL_PATH)
    if tok_result is not None:
        vocab, merge_priority, eos_id = tok_result
        # Verify vocab size matches
        max_id = max(vocab.values()) if vocab else 0
        if max_id >= VOCAB_SIZE:
            log(f"  HF tokenizer vocab ({max_id+1}) too large for "
                f"VOCAB_SIZE={VOCAB_SIZE}, retraining BPE")
            tok_result = None

    if tok_result is None:
        # Train BPE on subset of texts
        sample_size = min(len(texts), 10000)
        sample = random.sample(texts, sample_size) if len(texts) > \
            sample_size else texts
        vocab, merge_priority, eos_id = train_bpe_tokenizer(
            sample, VOCAB_SIZE)

    chunks = _tokenize_texts(texts, vocab, merge_priority, eos_id, seq_len)
    return chunks, has_real_data


# ── Main training ────────────────────────────────────────────────────


def main() -> None:
    t0_global = time.time()

    rank_id = int(os.environ.get("RANK_ID", "0"))
    rank_size = int(os.environ.get("RANK_SIZE", "1"))
    device_id = int(os.environ.get("DEVICE_ID", "0"))

    setup_logging(rank_id)

    log("=" * 60)
    log(f"Thinker-1.5B training | rank={rank_id}/{rank_size} "
        f"device={device_id}")
    log(f"Python {sys.version}")
    log(f"CWD: {os.getcwd()}")
    log("")
    log(f"CODE_PATH:     {CODE_PATH}")
    log(f"DATASET_PATH:  {DATASET_PATH}")
    log(f"PRETRAIN_PATH: {PRETRAIN_MODEL_PATH}")
    log(f"OUTPUT_PATH:   {OUTPUT_PATH}")
    log("")

    if os.path.isdir(DATASET_PATH):
        entries = sorted(os.listdir(DATASET_PATH))[:20]
        log(f"DATASET dir: {entries}")
    else:
        log("DATASET_PATH does not exist")

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
        device_id=device_id,
        memory_optimize_level="O1",
    )

    # Data parallel init
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
            log("Falling back to single-device training")
            use_dp = False
            rank_size = 1

    # Build model
    log("Building Thinker-1.5B model ...")
    model = Thinker15BModel()

    # Gradient checkpointing
    for i in range(NUM_LAYERS):
        getattr(model, f"layer{i}").recompute()

    total_p, trainable_p = count_params(model)
    log(f"Total params: {total_p:,}  |  Trainable: {trainable_p:,}")
    log(f"  Architecture: {NUM_LAYERS} layers, {HIDDEN_DIM}d, "
        f"{NUM_HEADS}Q/{NUM_KV_HEADS}KV GQA, {INTERMEDIATE_DIM} FFN")
    log(f"  SFM: {SFM_NUM_SLOTS} slots x {SFM_SLOT_DIM}d at layers "
        f"{SFM_LAYERS}")
    log(f"  Vocab: {VOCAB_SIZE}, SeqLen: {MAX_SEQ_LEN}, "
        f"Tied emb: {TIE_WORD_EMBEDDINGS}")

    # Load dataset
    log("Loading dataset ...")
    data_chunks, has_real_data = load_dataset(DATASET_PATH, MAX_SEQ_LEN)
    use_real_data = data_chunks is not None

    if use_real_data:
        log(f"Dataset: {data_chunks.shape[0]} chunks, "
            f"{data_chunks.shape[1]} tok/chunk, real_data={has_real_data}")
    else:
        log("WARNING: no dataset — using synthetic random tokens")

    # Precompute RoPE + causal mask (shared across all blocks)
    S = MAX_SEQ_LEN
    HD = HEAD_DIM
    inv_freq = 1.0 / (ROPE_THETA ** (
        np.arange(0, HD, 2, dtype=np.float32) / HD))
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
    log(f"RoPE + causal mask precomputed ({S}x{S})")

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
    log(f"LR schedule: {max_steps} steps, warmup={WARMUP_STEPS}, "
        f"lr=[{lr_schedule[0]:.2e}, {max(lr_schedule):.2e}, "
        f"{lr_schedule[-1]:.2e}]")

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

    # Batch size fallback (OOM recovery)
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

    # Training loop
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
    log("TRAINING START")
    log("=" * 60)
    if use_real_data:
        log(f"Dataset: {total_samples:,} chunks, "
            f"real_data={has_real_data}")

    while step < max_steps:
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            stop_reason = f"hard time limit ({elapsed:.0f}s > {TIME_LIMIT}s)"
            log(f"Time limit reached — {stop_reason}")
            break

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

        # Epoch tracking
        if use_real_data and samples_seen >= total_samples:
            epochs_completed += 1
            samples_seen -= total_samples
            log(f"=== Epoch {epochs_completed} completed at step {step} ===")

        # Convergence check (only after warmup and min epochs)
        loss_history.append(loss_float)
        if (len(loss_history) >= CONVERGENCE_WINDOW
                and step > WARMUP_STEPS
                and epochs_completed >= MIN_EPOCHS):
            rolling_avg = sum(loss_history[-CONVERGENCE_WINDOW:]) / \
                CONVERGENCE_WINDOW
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
                               f"{steps_without_improvement} steps, "
                               f"{epochs_completed} epochs done)")
                log(f"STOPPING: {stop_reason}")
                break

        # Logging
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

        # Checkpointing
        if rank_id == 0 and step % 1000 == 0 and step > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"step_{step}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            log(f"Checkpoint saved: {ckpt_path}")

    # Training complete
    elapsed_total = time.time() - t_start
    tps_avg = total_tokens / elapsed_total if elapsed_total > 0 else 0

    if rank_id == 0:
        ms.save_checkpoint(model, os.path.join(CKPT_DIR, "final.ckpt"))
        log("Final checkpoint saved")

    results = {
        "model_name": "Thinker-1.5B (from-scratch + SFM GQA)",
        "total_params": total_p,
        "trainable_params": trainable_p,
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
        "architecture": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "intermediate_dim": INTERMEDIATE_DIM,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "tied_embeddings": TIE_WORD_EMBEDDINGS,
        },
        "sfm_config": {
            "num_slots": SFM_NUM_SLOTS,
            "slot_dim": SFM_SLOT_DIM,
            "slot_heads": SFM_NUM_HEADS,
            "sfm_layers": list(SFM_LAYERS),
        },
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
            log("upload_output() completed successfully")
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
