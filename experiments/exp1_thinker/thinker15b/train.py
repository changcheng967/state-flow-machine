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
    dense.weight.set_data(Tensor(w))
    if has_bias:
        dense.bias.set_data(Tensor(np.zeros(out_ch, dtype=np.float16)))
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

        # Gated write-back: mean(slot_out) -> residual (broadcast (B,1,H))
        slot_mean = self.mean(new_slots, 1)  # (B, 1, SD)
        writeback = self.write_proj(slot_mean)  # (B, 1, H)
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
        self.tile = ops.Tile()

        # Initial slots — learned parameter (broadcast to batch)
        self.initial_slots = ms.Parameter(
            Tensor(np.random.randn(1, SFM_NUM_SLOTS, SFM_SLOT_DIM).astype(
                np.float16) * 0.02),
            name="initial_slots")

        # 24 transformer layers
        self.layers = nn.CellList(
            [TransformerBlock() for _ in range(NUM_LAYERS)])

        # 4 SFM banks at layers 5, 11, 17, 23
        self.sfm_banks = nn.CellList(
            [SFMSlotBank() for _ in range(len(SFM_LAYERS))])
        self.sfm_layer_set = set(SFM_LAYERS)

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
            x = self.layers[i](x, cos, sin, mask)
            if i in self.sfm_layer_set:
                x, slots = self.sfm_banks[sfm_idx](x, slots)
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
        self.cos = ms.Parameter(cos, name="cos", requires_grad=False)
        self.sin = ms.Parameter(sin, name="sin", requires_grad=False)
        self.mask = ms.Parameter(mask, name="mask", requires_grad=False)

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
    special_tokens = ["<pad>", "<eos>", "<unk>"] + SPECIAL_TOKENS
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


# ── Synthetic data generation (Phi-1 / SemCoder inspired) ────────────

# Special tokens for structured data
SPECIAL_TOKENS = [
    "<|code|>", "<|trace|>", "<|answer|>", "<|concept|>",
    "<|explanation|>", "<|usage|>", "<|buggy_code|>", "<|error|>",
    "<|reasoning|>", "<|fix|>", "<|documentation|>", "<|task|>",
    "<|think|>", "<|verify|>",
]

# Restricted builtins for exec()
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range,
    "int": int, "float": float, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "set": set, "bool": bool,
    "sum": sum, "round": round, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": filter,
    "isinstance": isinstance, "print": lambda *a: None,
    "True": True, "False": False, "None": None,
    "type": type, "hash": hash, "id": id, "hex": hex, "oct": oct,
    "bin": bin, "chr": chr, "ord": ord, "pow": pow, "divmod": divmod,
    "all": all, "any": any, "frozenset": frozenset,
}


def _exec_with_trace(code_str: str, timeout_s: float = 1.0):
    """Execute code with sys.settrace to capture line-by-line state.

    Returns (line_states, final_vars) or None on failure/timeout.
    line_states: list of (line_no, {var: repr(val)})
    """
    import signal as _sig

    namespace = {}
    line_states = []
    code_lines = code_str.strip().split("\n")

    def trace_fn(frame, event, arg):
        if event == "line":
            fname = frame.f_code.co_filename
            if fname == "<string>":
                lineno = frame.f_lineno
                snapshot = {}
                for k, v in frame.f_locals.items():
                    try:
                        snapshot[k] = repr(v)
                    except Exception:
                        snapshot[k] = "<error>"
                line_states.append((lineno, snapshot))
        return trace_fn

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    old_handler = None
    try:
        old_handler = _sig.signal(_sig.SIGALRM, _timeout_handler)
        _sig.alarm(int(timeout_s) + 1)
        sys.settrace(trace_fn)
        exec(code_str, {"__builtins__": _SAFE_BUILTINS}, namespace)
        sys.settrace(None)
        _sig.alarm(0)
    except Exception:
        return None
    finally:
        sys.settrace(None)
        if old_handler is not None:
            try:
                _sig.signal(_sig.SIGALRM, old_handler)
            except Exception:
                pass

    # Build final answer from last line state
    final_vars = {}
    if line_states:
        final_vars = dict(line_states[-1][1])

    return line_states, final_vars


def _gen_random_program(complexity: int) -> str:
    """Generate a random Python program with given complexity (1-5).

    1: simple assignments + arithmetic
    2: if/else, simple loops
    3: nested loops, functions
    4: recursion, list operations, dict usage
    5: multi-function programs with list comprehensions
    """
    vars_pool = ["x", "y", "z", "n", "i", "j", "k", "a", "b", "c",
                  "total", "result", "count", "found", "data", "tmp"]
    lines = []

    if complexity == 1:
        n_vars = random.randint(2, 5)
        used = random.sample(vars_pool, n_vars)
        for v in used:
            lines.append(f"{v} = {random.randint(-50, 50)}")
        for _ in range(random.randint(1, 4)):
            v1 = random.choice(used)
            op = random.choice(["+", "-", "*"])
            v2 = random.choice(used) if random.random() < 0.7 else \
                random.randint(-20, 20)
            lines.append(f"{v1} = {v1} {op} {v2}")

    elif complexity == 2:
        kind = random.choice(["if", "for", "while"])
        if kind == "if":
            n = random.randint(1, 30)
            lines.append(f"n = {n}")
            lines.append(f"if n > {random.randint(10, 20)}:")
            lines.append(f"    result = n * 2")
            lines.append(f"else:")
            lines.append(f"    result = n + {random.randint(1, 5)}")
        elif kind == "for":
            n = random.randint(3, 15)
            lines.append(f"total = 0")
            lines.append(f"for i in range({n}):")
            op = random.choice(["+", "*"])
            val = random.randint(1, 5)
            lines.append(f"    total = total {op} {val}")
        elif kind == "while":
            x0 = random.randint(0, 10)
            target = random.randint(15, 50)
            lines.append(f"x = {x0}")
            lines.append(f"while x < {target}:")
            lines.append(f"    x = x + {random.randint(1, 5)}")

    elif complexity == 3:
        kind = random.choice(["nested", "func_simple", "list_ops"])
        if kind == "nested":
            n = random.randint(3, 8)
            lines.append(f"result = 0")
            lines.append(f"for i in range({n}):")
            cond = random.choice(["i % 2 == 0", "i % 3 == 0",
                                   "i > 0", "True"])
            op = random.choice(["+", "-"])
            val = random.randint(1, 5)
            lines.append(f"    if {cond}:")
            lines.append(f"        result = result {op} {val}")
        elif kind == "func_simple":
            fn = random.choice(["double", "negate", "clamp"])
            if fn == "double":
                lines.append("def double(x):")
                lines.append("    return x * 2")
            elif fn == "negate":
                lines.append("def negate(x):")
                lines.append("    return -x")
            else:
                lines.append("def clamp(x, lo=0, hi=100):")
                lines.append("    if x < lo:")
                lines.append("        return lo")
                lines.append("    if x > hi:")
                lines.append("        return hi")
                lines.append("    return x")
            val = random.randint(1, 50)
            lines.append(f"result = {fn}({val})")
        elif kind == "list_ops":
            arr = [random.randint(1, 20) for _ in range(random.randint(3, 8))]
            lines.append(f"data = {arr}")
            op = random.choice(["sum", "max", "min", "len", "sorted"])
            lines.append(f"result = {op}(data)")

    elif complexity == 4:
        kind = random.choice(["recursive", "dict_ops", "list_build"])
        if kind == "recursive":
            fn = random.choice(["fib", "fact", "pow2"])
            if fn == "fib":
                lines.append("def fib(n):")
                lines.append("    if n <= 1:")
                lines.append("        return n")
                lines.append("    return fib(n-1) + fib(n-2)")
                lines.append(f"result = fib({random.randint(4, 10)})")
            elif fn == "fact":
                lines.append("def fact(n):")
                lines.append("    if n <= 1:")
                lines.append("        return 1")
                lines.append("    return n * fact(n-1)")
                lines.append(f"result = fact({random.randint(3, 8)})")
            else:
                lines.append("def pow2(n):")
                lines.append("    if n == 0:")
                lines.append("        return 1")
                lines.append("    return 2 * pow2(n-1)")
                lines.append(f"result = pow2({random.randint(2, 8)})")
        elif kind == "dict_ops":
            lines.append("d = {}")
            keys = ["a", "b", "c", "x", "y"]
            n_pairs = random.randint(2, 4)
            for k in keys[:n_pairs]:
                lines.append(f"d['{k}'] = {random.randint(1, 100)}")
            lines.append(f"result = sum(d.values())")
        elif kind == "list_build":
            lines.append(f"result = []")
            n = random.randint(3, 8)
            lines.append(f"for i in range({n}):")
            expr = random.choice(["i*i", "i*2+1", "i if i%2==0 else 0"])
            lines.append(f"    result.append({expr})")

    elif complexity == 5:
        kind = random.choice(["comp", "multi_func", "filter_map"])
        if kind == "comp":
            arr = [random.randint(1, 30) for _ in range(random.randint(4, 10))]
            pred = random.choice(["x % 2 == 0", "x > 10", "x % 3 == 0"])
            expr = random.choice(["x*x", "x*2", "x+1"])
            lines.append(f"data = {arr}")
            lines.append(f"result = [{expr} for x in data if {pred}]")
        elif kind == "multi_func":
            lines.append("def square(x):")
            lines.append("    return x * x")
            lines.append("")
            lines.append("def sum_sq(lst):")
            lines.append("    total = 0")
            lines.append("    for v in lst:")
            lines.append("        total = total + square(v)")
            lines.append("    return total")
            lines.append("")
            arr = [random.randint(1, 10) for _ in range(random.randint(3, 6))]
            lines.append(f"result = sum_sq({arr})")
        elif kind == "filter_map":
            arr = [random.randint(-10, 30) for _ in range(random.randint(4, 8))]
            lines.append(f"data = {arr}")
            lines.append("pos = [x for x in data if x > 0]")
            lines.append("result = [x*2 for x in pos]")

    return "\n".join(lines)


def _stage1_mental_simulation(n_samples: int, seed: int) -> list:
    """Stage 1 (30%): Mental simulation — exec code + capture trace.

    Teaches the model to simulate execution step by step, the core
    skill transformers provably lack.
    """
    random.seed(seed)
    texts = []
    success = 0

    for _ in range(n_samples * 3):  # Over-generate; many will fail
        if success >= n_samples:
            break
        complexity = random.choices(
            range(1, 6), weights=[15, 30, 25, 20, 10])[0]
        code = _gen_random_program(complexity)
        if not code.strip():
            continue

        result = _exec_with_trace(code, timeout_s=1.0)
        if result is None:
            continue

        line_states, final_vars = result
        if not line_states:
            continue

        # Format: <|code|> ... <|trace|> ... <|answer|> ...
        parts = []
        parts.append("<|code|>")
        parts.append(code)
        parts.append("<|trace|>")
        for lineno, snapshot in line_states:
            line_text = code_lines_get(code, lineno)
            state_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(snapshot.items()))
            parts.append(f"Line {lineno}: {line_text} -> state: {{{state_str}}}")
        parts.append("<|answer|>")
        # Pick the last assigned variable as answer
        last_line = code.strip().split("\n")[-1].strip()
        if "=" in last_line and not last_line.startswith("def ") and \
                not last_line.startswith("for ") and \
                not last_line.startswith("while ") and \
                not last_line.startswith("if "):
            var = last_line.split("=")[0].strip()
            if var in final_vars:
                parts.append(f"Final: {var} = {final_vars[var]}")
            else:
                parts.append(f"Final state: {dict(sorted(final_vars.items()))}")
        else:
            parts.append(f"Final state: {dict(sorted(final_vars.items()))}")

        texts.append("\n".join(parts))
        success += 1

    log(f"  Stage 1 (Mental Simulation): {success}/{n_samples*3} succeeded")
    return texts


def code_lines_get(code: str, lineno: int) -> str:
    """Get a line from code by 1-indexed line number."""
    lines = code.split("\n")
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return "..."


def _stage2_textbook_explanations(n_samples: int, seed: int) -> list:
    """Stage 2 (25%): Textbook-quality explanations.

    Each sample teaches a concept with explanation + code + usage.
    Progressive difficulty within each concept.
    """
    random.seed(seed)

    concepts = {
        "Variables": [
            ("A variable stores a value in memory. You assign with '='.",
             "x = 42\nprint(x)", "x -> 42"),
            ("Variables can be reassigned. Python is dynamically typed.",
             "x = 10\nx = 'hello'\nprint(x)", "x is now the string 'hello'"),
            ("Multiple assignment swaps values without a temp variable.",
             "a, b = 3, 7\na, b = b, a\nprint(a, b)", "a=7, b=3"),
            ("Use descriptive variable names for readability.",
             "user_age = 25\nmax_connections = 100\nprint(user_age * 2)",
             "50"),
        ],
        "Operators": [
            ("Arithmetic operators: + - * // % **. Note // is floor division.",
             "print(17 // 3)\nprint(17 % 3)\nprint(2 ** 10)",
             "5, 2, 1024"),
            ("Comparison operators return bool: == != < > <= >=",
             "x = 15\nprint(x > 10)\nprint(x == 15)\nprint(x != 20)",
             "True, True, True"),
            ("Logical operators: and, or, not. Short-circuit evaluation.",
             "x = 5\nprint(x > 0 and x < 10)\nprint(x < 0 or x > 3)",
             "True, True"),
            ("The 'in' operator tests membership in sequences.",
             "print(3 in [1, 2, 3, 4])\nprint('h' in 'hello')",
             "True, True"),
        ],
        "Control Flow": [
            ("if/elif/else executes blocks based on conditions.",
             "score = 85\nif score >= 90:\n    grade = 'A'\nelif score >= 80:\n    grade = 'B'\nelse:\n    grade = 'C'\nprint(grade)", "B"),
            ("for loops iterate over any iterable (range, list, string).",
             "total = 0\nfor i in range(1, 6):\n    total += i\nprint(total)",
             "15 (sum of 1+2+3+4+5)"),
            ("while loops repeat until a condition is false.",
             "n = 64\nsteps = 0\nwhile n > 1:\n    n = n // 2\n    steps += 1\nprint(steps)",
             "6 (64->32->16->8->4->2->1)"),
            ("break exits a loop early. continue skips to next iteration.",
             "for i in range(10):\n    if i == 5:\n        break\n    if i % 2 == 0:\n        continue\n    print(i)",
             "1, 3"),
            ("Nested loops: inner loop runs fully for each outer iteration.",
             "for i in range(3):\n    for j in range(3):\n        if i == j:\n            print(i)",
             "0, 1, 2 (diagonal)"),
        ],
        "Functions": [
            ("A function encapsulates reusable logic with def.",
             "def greet(name):\n    return f'Hello, {name}!'\nprint(greet('World'))",
             "Hello, World!"),
            ("Default parameters make arguments optional.",
             "def power(base, exp=2):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result\nprint(power(3))\nprint(power(2, 10))",
             "9, 1024"),
            ("*args collects positional args into a tuple.",
             "def total(*nums):\n    return sum(nums)\nprint(total(1, 2, 3, 4, 5))",
             "15"),
            ("Recursion: a function calls itself with a smaller input.",
             "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\nprint(gcd(48, 18))",
             "6"),
        ],
        "Data Structures": [
            ("Lists are ordered mutable sequences. Index from 0.",
             "fruits = ['apple', 'banana', 'cherry']\nfruits.append('date')\nprint(fruits[1])\nprint(len(fruits))",
             "banana, 4"),
            ("List slicing: lst[start:stop:step]. Stop is exclusive.",
             "nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\nprint(nums[2:7])\nprint(nums[::2])\nprint(nums[::-1])",
             "[2,3,4,5,6], [0,2,4,6,8], [9,8,7,6,5,4,3,2,1,0]"),
            ("Dicts store key-value pairs. O(1) lookup by key.",
             "ages = {'Alice': 30, 'Bob': 25, 'Charlie': 35}\nages['Dave'] = 28\nprint(sorted(ages.items()))",
             "[('Alice', 30), ('Bob', 25), ('Charlie', 35), ('Dave', 28)]"),
            ("Sets store unique elements. Support union, intersection, diff.",
             "a = {1, 2, 3, 4}\nb = {3, 4, 5, 6}\nprint(a & b)\nprint(a | b)\nprint(a - b)",
             "{3, 4}, {1, 2, 3, 4, 5, 6}, {1, 2}"),
            ("List comprehensions: concise way to create lists.",
             "squares = [x*x for x in range(1, 6)]\nevens = [x for x in range(20) if x % 2 == 0]\nprint(squares)\nprint(evens[:5])",
             "[1, 4, 9, 16, 25], [0, 2, 4, 6, 8]"),
        ],
        "Sorting & Searching": [
            ("Binary search finds a target in a sorted array in O(log n).",
             "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nprint(binary_search([1, 3, 5, 7, 9, 11], 7))",
             "3 (index of 7)"),
            ("Bubble sort repeatedly swaps adjacent out-of-order elements.",
             "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(n - i - 1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\nprint(bubble_sort([5, 3, 8, 1, 9, 2]))",
             "[1, 2, 3, 5, 8, 9]"),
            ("Two-pointer technique: sort then find pairs meeting a condition.",
             "def two_sum_sorted(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        s = arr[lo] + arr[hi]\n        if s == target:\n            return [lo, hi]\n        elif s < target:\n            lo += 1\n        else:\n            hi -= 1\n    return None\nprint(two_sum_sorted([1, 3, 5, 7, 11], 12))",
             "[1, 3] (3 + 9 = 12)"),
        ],
        "String Operations": [
            ("Strings are immutable sequences. Use + to concatenate.",
             "first = 'hello'\nsecond = ' world'\nresult = first + second\nprint(len(result))\nprint(result.upper())",
             "11, HELLO WORLD"),
            ("str.split() and str.join() for parsing/formatting.",
             "line = 'name:Alice,age:30,city:NYC'\nparts = line.split(',')\nresult = '; '.join(parts)\nprint(result)",
             "name:Alice; age:30; city:NYC"),
            ("f-strings interpolate variables directly into strings.",
             "name = 'Alice'\nscore = 95\nresult = f'{name} scored {score}/100'\nprint(result)",
             "Alice scored 95/100"),
            ("str.strip(), str.replace(), str.count() for cleaning.",
             "text = '  hello world  '\nprint(text.strip())\nprint(text.strip().replace('world', 'Python'))",
             "hello world, hello Python"),
        ],
        "Error Handling": [
            ("try/except catches exceptions so the program doesn't crash.",
             "def safe_divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None\nprint(safe_divide(10, 3))\nprint(safe_divide(10, 0))",
             "3.333..., None"),
            ("finally block always runs, even if an exception occurred.",
             "def process(data):\n    try:\n        result = int(data)\n    except ValueError:\n        result = 0\n    finally:\n        print('done')\n    return result\nprint(process('42'))\nprint(process('abc'))",
             "done, 42, done, 0"),
        ],
        "Recursion": [
            ("Factorial: n! = n * (n-1)!. Base case: 0! = 1.",
             "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\nprint(factorial(5))\nprint(factorial(10))",
             "120, 3628800"),
            ("Fibonacci: each number is the sum of the two before it.",
             "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\nprint(fib(0))\nprint(fib(7))",
             "0, 13"),
            ("Tower of Hanoi: move n disks from source to target using aux.",
             "def hanoi(n, src, dst, aux):\n    if n == 1:\n        return 1\n    return hanoi(n-1, src, aux, dst) + 1 + hanoi(n-1, aux, dst, src)\nprint(hanoi(3))\nprint(hanoi(5))",
             "7, 31"),
        ],
        "OOP Basics": [
            ("A class defines a blueprint. __init__ sets up instance state.",
             "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n    def increment(self):\n        self.value += 1\n    def get(self):\n        return self.value\nc = Counter(10)\nc.increment()\nc.increment()\nprint(c.get())",
             "12"),
            ("Inheritance: a subclass extends a parent class.",
             "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        return '...'\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'\nd = Dog('Rex')\nprint(d.speak())",
             "Rex says Woof!"),
        ],
    }

    texts = []
    for concept_name, examples in concepts.items():
        for explanation, code, expected in examples:
            parts = []
            parts.append("<|concept|> " + concept_name)
            parts.append("<|explanation|> " + explanation)
            parts.append("<|code|>")
            parts.append(code)
            parts.append("<|usage|> " + expected)
            texts.append("\n".join(parts))

    # Duplicate to reach target
    while len(texts) < n_samples:
        extra = list(texts)
        random.shuffle(extra)
        texts.extend(extra)
    random.shuffle(texts)
    return texts[:n_samples]


def _stage3_monologue_debugging(n_samples: int, seed: int) -> list:
    """Stage 3 (20%): Buggy code + step-by-step reasoning to fix.

    SemCoder-inspired: the model narrates its debugging process.
    """
    random.seed(seed)

    bug_templates = [
        {
            "buggy": "def sum_evens(lst):\n    total = 0\n    for i in range(len(lst)):\n        if lst[i] % 2 == 0:\n            total += lst[i]\n        return total",
            "error": "sum_evens([1, 2, 3, 4]) returns 0 instead of 6",
            "reasoning": "Let me trace: i=0, lst[0]=1, 1%2!=0, skip. "
                         "Then return total -> returns 0! The return is "
                         "inside the for loop (wrong indentation). It returns "
                         "after the first element instead of after all.",
            "fix": "def sum_evens(lst):\n    total = 0\n    for i in range(len(lst)):\n        if lst[i] % 2 == 0:\n            total += lst[i]\n    return total",
        },
        {
            "buggy": "def find_max(lst):\n    max_val = lst[0]\n    for i in range(1, len(lst) + 1):\n        if lst[i] > max_val:\n            max_val = lst[i]\n    return max_val",
            "error": "find_max([3, 7, 2, 9]) raises IndexError",
            "reasoning": "The range goes to len(lst) which is 4. lst[4] "
                         "is out of bounds. The range should be range(1, "
                         "len(lst)), not len(lst) + 1. Off-by-one error.",
            "fix": "def find_max(lst):\n    max_val = lst[0]\n    for i in range(1, len(lst)):\n        if lst[i] > max_val:\n            max_val = lst[i]\n    return max_val",
        },
        {
            "buggy": "def reverse_str(s):\n    result = ''\n    for i in range(len(s)):\n        result = s[i] + result\n    return result\nprint(reverse_str('hello'))",
            "error": "reverse_str('hello') returns 'olleh' — wait, "
                     "that's correct actually. Let me check again.",
            "reasoning": "Actually this IS correct: prepending each char "
                         "builds the reversed string. The 'bug' here is "
                         "subtle — the function works but is O(n^2) due "
                         "to string concatenation in a loop. Better: s[::-1].",
            "fix": "def reverse_str(s):\n    return s[::-1]",
        },
        {
            "buggy": "def is_palindrome(s):\n    return s == s.reverse()",
            "error": "is_palindrome('racecar') raises AttributeError",
            "reasoning": "str.reverse() doesn't exist — that's a list method. "
                         "Strings are immutable. Should use slicing: "
                         "s[::-1] or reversed(s).",
            "fix": "def is_palindrome(s):\n    return s == s[::-1]",
        },
        {
            "buggy": "def count_words(text):\n    words = text.split()\n    for word in words:\n        count = 1\n    return count",
            "error": "count_words('hello world foo') returns 1 instead of 3",
            "reasoning": "count is initialized INSIDE the loop, so it resets "
                         "to 1 on every iteration. Should initialize before "
                         "the loop and increment, or just use len(words).",
            "fix": "def count_words(text):\n    return len(text.split())",
        },
        {
            "buggy": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) or j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result",
            "error": "merge_sorted([1, 3], [2, 4]) raises IndexError",
            "reasoning": "When i reaches len(a), a[i] is out of bounds. "
                         "Need to check bounds before comparing. Also need "
                         "to handle remaining elements when one list is "
                         "exhausted.",
            "fix": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
        },
        {
            "buggy": "def remove_dupes(lst):\n    seen = set()\n    for x in lst:\n        if x not in seen:\n            seen.append(x)\n    return list(seen)",
            "error": "remove_dupes([1,2,2,3]) raises AttributeError",
            "reasoning": "Sets don't have an append() method. Use .add() "
                         "for sets, or use a list for seen.",
            "fix": "def remove_dupes(lst):\n    seen = []\n    for x in lst:\n        if x not in seen:\n            seen.append(x)\n    return seen",
        },
        {
            "buggy": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1",
            "error": "binary_search([1, 3, 5], 5) returns -1",
            "reasoning": "The condition should be 'while lo <= hi', not "
                         "'while lo < hi'. When lo == hi, the last element "
                         "might be the target, but the loop exits early.",
            "fix": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        },
        {
            "buggy": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            for sub in item:\n                result.append(sub)\n    return result",
            "error": "flatten([1, [2, 3], 4]) returns [2, 3] instead of "
                     "[1, 2, 3, 4]",
            "reasoning": "Non-list items are never appended! The else "
                         "branch is missing. Only list items get processed.",
            "fix": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(item)\n        else:\n            result.append(item)\n    return result",
        },
        {
            "buggy": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            "error": "fibonacci(0) returns 1 instead of 0",
            "reasoning": "The loop runs n times even when n=0. Actually "
                         "when n=0, range(0) is empty so a=0 is returned "
                         "— wait, that IS correct. Let me re-check... "
                         "fib(1) should be 1. After 1 iteration: a=1, b=1. "
                         "Returns 1. Correct. fib(6) = 8? Let me trace: "
                         "iter1: a=1,b=1 iter2: a=1,b=2 iter3: a=2,b=3 "
                         "iter4: a=3,b=5 iter5: a=5,b=8 iter6: a=8,b=13. "
                         "Returns 8. Correct! The bug report was wrong.",
            "fix": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a  # No bug — this is correct!",
        },
    ]

    texts = []
    for tmpl in bug_templates:
        parts = []
        parts.append("<|buggy_code|>")
        parts.append(tmpl["buggy"])
        parts.append("<|error|>")
        parts.append(tmpl["error"])
        parts.append("<|reasoning|>")
        parts.append(tmpl["reasoning"])
        parts.append("<|fix|>")
        parts.append(tmpl["fix"])
        texts.append("\n".join(parts))

    # Duplicate with variation to reach target
    while len(texts) < n_samples:
        extra = []
        for tmpl in bug_templates:
            parts = []
            parts.append("<|buggy_code|>")
            parts.append(tmpl["buggy"])
            parts.append("<|error|>")
            parts.append(tmpl["error"])
            parts.append("<|reasoning|>")
            parts.append(tmpl["reasoning"])
            parts.append("<|fix|>")
            parts.append(tmpl["fix"])
            extra.append("\n".join(parts))
        random.shuffle(extra)
        texts.extend(extra)
    random.shuffle(texts)
    return texts[:n_samples]


def _stage4_novel_api(n_samples: int, seed: int) -> list:
    """Stage 4 (15%): Fictional API documentation + usage task.

    Teaches learning-from-context — reading docs to use novel APIs.
    """
    random.seed(seed)

    api_templates = [
        {
            "doc": (
                "class DataPipeline:\n"
                "    \"\"\"Process data through a sequence of transformations.\"\"\"\n"
                "    def __init__(self, source: str):\n"
                "        \"\"\"Initialize pipeline with data source path.\"\"\"\n"
                "        self.steps = []\n"
                "        self.source = source\n"
                "    def add_step(self, fn: callable, name: str = '') -> 'DataPipeline':\n"
                "        \"\"\"Add a transformation step. Returns self for chaining.\"\"\"\n"
                "        self.steps.append((name, fn))\n"
                "        return self\n"
                "    def run(self, limit: int = -1) -> list:\n"
                "        \"\"\"Execute pipeline. limit=-1 means process all.\"\"\"\n"
                "        data = [1, 2, 3, 4, 5]  # simulated from source\n"
                "        for name, fn in self.steps:\n"
                "            data = [fn(x) for x in data]\n"
                "        if limit > 0:\n"
                "            data = data[:limit]\n"
                "        return data"
            ),
            "task": "Create a DataPipeline from 'data.csv' that squares each "
                    "value, then keeps only values greater than 5.",
            "solution": (
                "pipeline = DataPipeline('data.csv')\n"
                "pipeline.add_step(lambda x: x * x, 'square')\n"
                "pipeline.add_step(lambda x: x if x > 5 else None, 'filter')\n"
                "result = pipeline.run()"
            ),
        },
        {
            "doc": (
                "class TimeSeries:\n"
                "    \"\"\"Analyze a sequence of numerical values over time.\"\"\"\n"
                "    def __init__(self, values: list):\n"
                "        self.values = values\n"
                "    def moving_avg(self, window: int) -> list:\n"
                "        \"\"\"Compute moving average with given window size.\"\"\"\n"
                "        result = []\n"
                "        for i in range(len(self.values) - window + 1):\n"
                "            chunk = self.values[i:i+window]\n"
                "            result.append(sum(chunk) / window)\n"
                "        return result\n"
                "    def trend(self) -> str:\n"
                "        \"\"\"Return 'up', 'down', or 'flat' based on slope.\"\"\"\n"
                "        if len(self.values) < 2:\n"
                "            return 'flat'\n"
                "        diff = self.values[-1] - self.values[0]\n"
                "        if diff > 0:\n"
                "            return 'up'\n"
                "        elif diff < 0:\n"
                "            return 'down'\n"
                "        return 'flat'"
            ),
            "task": "Create a TimeSeries with values [10, 15, 12, 20, 18, 25], "
                    "compute the 3-period moving average, and determine the trend.",
            "solution": (
                "ts = TimeSeries([10, 15, 12, 20, 18, 25])\n"
                "avg = ts.moving_avg(3)\n"
                "direction = ts.trend()\n"
                "print(avg)  # [12.33, 15.67, 16.67, 21.0]\n"
                "print(direction)  # 'up'"
            ),
        },
        {
            "doc": (
                "class RateLimiter:\n"
                "    \"\"\"Limit how many operations can occur in a time window.\"\"\"\n"
                "    def __init__(self, max_ops: int, window: int):\n"
                "        self.max_ops = max_ops\n"
                "        self.window = window\n"
                "        self.ops = []\n"
                "    def allow(self, timestamp: int) -> bool:\n"
                "        \"\"\"Check if operation is allowed at given timestamp.\"\"\"\n"
                "        cutoff = timestamp - self.window\n"
                "        self.ops = [t for t in self.ops if t > cutoff]\n"
                "        if len(self.ops) < self.max_ops:\n"
                "            self.ops.append(timestamp)\n"
                "            return True\n"
                "        return False"
            ),
            "task": "Create a RateLimiter allowing 3 operations per 10-second "
                    "window. Test it at timestamps 0, 2, 4, 5, 15.",
            "solution": (
                "rl = RateLimiter(max_ops=3, window=10)\n"
                "print(rl.allow(0))   # True (1/3)\n"
                "print(rl.allow(2))   # True (2/3)\n"
                "print(rl.allow(4))   # True (3/3)\n"
                "print(rl.allow(5))   # False (4th op within window)\n"
                "print(rl.allow(15))  # True (old ops expired)"
            ),
        },
        {
            "doc": (
                "class SparseVector:\n"
                "    \"\"\"Efficiently represent vectors with mostly zero values.\"\"\"\n"
                "    def __init__(self, dim: int):\n"
                "        self.dim = dim\n"
                "        self.data = {}  # {index: value}\n"
                "    def set(self, idx: int, val: float):\n"
                "        \"\"\"Set value at index.\"\"\"\n"
                "        if val != 0:\n"
                "            self.data[idx] = val\n"
                "        elif idx in self.data:\n"
                "            del self.data[idx]\n"
                "    def dot(self, other: 'SparseVector') -> float:\n"
                "        \"\"\"Dot product with another sparse vector.\"\"\"\n"
                "        result = 0.0\n"
                "        for idx, val in self.data.items():\n"
                "            if idx in other.data:\n"
                "                result += val * other.data[idx]\n"
                "        return result"
            ),
            "task": "Create two SparseVectors of dimension 100. Set v1 at "
                    "indices 0,5,10 with values 1,2,3. Set v2 at indices "
                    "5,10,15 with values 4,5,6. Compute their dot product.",
            "solution": (
                "v1 = SparseVector(100)\n"
                "v1.set(0, 1)\nv1.set(5, 2)\nv1.set(10, 3)\n"
                "v2 = SparseVector(100)\n"
                "v2.set(5, 4)\nv2.set(10, 5)\nv2.set(15, 6)\n"
                "result = v1.dot(v2)  # 2*4 + 3*5 = 23\n"
                "print(result)"
            ),
        },
        {
            "doc": (
                "class EventBus:\n"
                "    \"\"\"Pub/sub event system for decoupled communication.\"\"\"\n"
                "    def __init__(self):\n"
                "        self.subscribers = {}\n"
                "    def subscribe(self, event: str, handler: callable):\n"
                "        \"\"\"Register a handler for an event.\"\"\"\n"
                "        if event not in self.subscribers:\n"
                "            self.subscribers[event] = []\n"
                "        self.subscribers[event].append(handler)\n"
                "    def emit(self, event: str, data):\n"
                "        \"\"\"Emit event, calling all subscribers.\"\"\"\n"
                "        for handler in self.subscribers.get(event, []):\n"
                "            handler(data)\n"
                "    def on(self, event: str):\n"
                "        \"\"\"Decorator to register handler.\"\"\"\n"
                "        def decorator(fn):\n"
                "            self.subscribe(event, fn)\n"
                "            return fn\n"
                "        return decorator"
            ),
            "task": "Create an EventBus, subscribe to 'user_created' event "
                    "with a handler that prints the username, and emit the "
                    "event with {'username': 'alice'}.",
            "solution": (
                "bus = EventBus()\n"
                "def on_user_created(data):\n"
                "    print(f'New user: {data[\"username\"]}')\n"
                "bus.subscribe('user_created', on_user_created)\n"
                "bus.emit('user_created', {'username': 'alice'})\n"
                "# Output: New user: alice"
            ),
        },
    ]

    texts = []
    for tmpl in api_templates:
        parts = []
        parts.append("<|documentation|>")
        parts.append(tmpl["doc"])
        parts.append("<|task|>")
        parts.append(tmpl["task"])
        parts.append("<|solution|>")
        parts.append(tmpl["solution"])
        texts.append("\n".join(parts))

    while len(texts) < n_samples:
        extra = []
        for tmpl in api_templates:
            parts = []
            parts.append("<|documentation|>")
            parts.append(tmpl["doc"])
            parts.append("<|task|>")
            parts.append(tmpl["task"])
            parts.append("<|solution|>")
            parts.append(tmpl["solution"])
            extra.append("\n".join(parts))
        random.shuffle(extra)
        texts.extend(extra)
    random.shuffle(texts)
    return texts[:n_samples]


def _stage5_agent_trajectories(n_samples: int, seed: int) -> list:
    """Stage 5 (10%): Multi-step reasoning chains for coding tasks.

    Simulates how an agent thinks through a problem before coding.
    """
    random.seed(seed)

    tasks = [
        {
            "task": "Write a function that finds the longest common "
                   "subsequence of two strings.",
            "think": (
                "This is a classic dynamic programming problem.\n"
                "1. Create a 2D table dp of size (m+1) x (n+1)\n"
                "2. Fill bottom-up: if chars match, dp[i][j] = dp[i-1][j-1] + 1\n"
                "3. If chars differ, dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
                "4. Backtrack from dp[m][n] to reconstruct the subsequence"
            ),
            "code": (
                "def lcs(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            if s1[i-1] == s2[j-1]:\n"
                "                dp[i][j] = dp[i-1][j-1] + 1\n"
                "            else:\n"
                "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
                "    result = []\n"
                "    i, j = m, n\n"
                "    while i > 0 and j > 0:\n"
                "        if s1[i-1] == s2[j-1]:\n"
                "            result.append(s1[i-1])\n"
                "            i -= 1\n"
                "            j -= 1\n"
                "        elif dp[i-1][j] > dp[i][j-1]:\n"
                "            i -= 1\n"
                "        else:\n"
                "            j -= 1\n"
                "    return ''.join(reversed(result))"
            ),
            "verify": "lcs('ABCBDAB', 'BDCAB') -> 'BCAB'\n"
                     "lcs('', 'ABC') -> ''\n"
                     "lcs('ABC', 'ABC') -> 'ABC'",
        },
        {
            "task": "Write a function that validates parentheses in a string.",
            "think": (
                "Use a stack-based approach:\n"
                "1. Iterate through each character\n"
                "2. Push opening brackets onto stack\n"
                "3. For closing brackets, pop and check if types match\n"
                "4. If types don't match or stack empty when we need to pop: invalid\n"
                "5. At end, string is valid if stack is empty"
            ),
            "code": (
                "def is_valid(s):\n"
                "    stack = []\n"
                "    pairs = {')': '(', ']': '[', '}': '{'}\n"
                "    for ch in s:\n"
                "        if ch in '({[':\n"
                "            stack.append(ch)\n"
                "        elif ch in ')}]':\n"
                "            if not stack or stack[-1] != pairs[ch]:\n"
                "                return False\n"
                "            stack.pop()\n"
                "    return len(stack) == 0"
            ),
            "verify": "is_valid('()') -> True\n"
                     "is_valid('({[]})') -> True\n"
                     "is_valid('(]') -> False\n"
                     "is_valid('') -> True",
        },
        {
            "task": "Implement a min-heap with insert and extract_min.",
            "think": (
                "A min-heap is a complete binary tree where parent <= children.\n"
                "1. Use a list. Children of node i are at 2i+1 and 2i+2.\n"
                "2. Insert: append, then bubble up (swap with parent while < parent)\n"
                "3. Extract min: swap root with last, pop last, bubble down\n"
                "   (swap with smaller child while child < current)"
            ),
            "code": (
                "class MinHeap:\n"
                "    def __init__(self):\n"
                "        self.data = []\n"
                "    def insert(self, val):\n"
                "        self.data.append(val)\n"
                "        i = len(self.data) - 1\n"
                "        while i > 0:\n"
                "            parent = (i - 1) // 2\n"
                "            if self.data[i] < self.data[parent]:\n"
                "                self.data[i], self.data[parent] = self.data[parent], self.data[i]\n"
                "                i = parent\n"
                "            else:\n"
                "                break\n"
                "    def extract_min(self):\n"
                "        if not self.data:\n"
                "            return None\n"
                "        val = self.data[0]\n"
                "        self.data[0] = self.data[-1]\n"
                "        self.data.pop()\n"
                "        i = 0\n"
                "        while True:\n"
                "            left = 2*i + 1\n"
                "            right = 2*i + 2\n"
                "            smallest = i\n"
                "            if left < len(self.data) and self.data[left] < self.data[smallest]:\n"
                "                smallest = left\n"
                "            if right < len(self.data) and self.data[right] < self.data[smallest]:\n"
                "                smallest = right\n"
                "            if smallest == i:\n"
                "                break\n"
                "            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]\n"
                "            i = smallest\n"
                "        return val"
            ),
            "verify": (
                "h = MinHeap()\n"
                "h.insert(5)\nh.insert(3)\nh.insert(8)\nh.insert(1)\n"
                "print(h.extract_min())  # 1\n"
                "print(h.extract_min())  # 3\n"
                "print(h.extract_min())  # 5\n"
                "print(h.extract_min())  # 8"
            ),
        },
        {
            "task": "Write a function that counts the number of islands "
                   "in a 2D grid (DFS).",
            "think": (
                "An island is a group of adjacent '1's (4-directional).\n"
                "1. Iterate every cell in the grid\n"
                "2. When we find an unvisited '1', it's a new island\n"
                "3. DFS from that cell to mark all connected '1's as visited\n"
                "4. Increment island count, continue scanning"
            ),
            "code": (
                "def count_islands(grid):\n"
                "    if not grid:\n"
                "        return 0\n"
                "    rows, cols = len(grid), len(grid[0])\n"
                "    visited = set()\n"
                "    count = 0\n"
                "    def dfs(r, c):\n"
                "        stack = [(r, c)]\n"
                "        while stack:\n"
                "            cr, cc = stack.pop()\n"
                "            if (cr, cc) in visited:\n"
                "                continue\n"
                "            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:\n"
                "                continue\n"
                "            if grid[cr][cc] == 0:\n"
                "                continue\n"
                "            visited.add((cr, cc))\n"
                "            stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])\n"
                "    for r in range(rows):\n"
                "        for c in range(cols):\n"
                "            if grid[r][c] == 1 and (r, c) not in visited:\n"
                "                dfs(r, c)\n"
                "                count += 1\n"
                "    return count"
            ),
            "verify": (
                "count_islands([[1,1,0,0],[1,0,0,1],[0,0,1,1],[0,1,0,0]]) -> 3\n"
                "count_islands([[1,1,1],[1,1,1]]) -> 1\n"
                "count_islands([[0,0,0],[0,0,0]]) -> 0"
            ),
        },
        {
            "task": "Implement an LRU (Least Recently Used) cache.",
            "think": (
                "LRU evicts the least recently accessed item when full.\n"
                "1. Use OrderedDict (or dict + doubly linked list)\n"
                "2. get(key): move to front (most recent), return value\n"
                "3. put(key, value): if exists, update and move to front\n"
                "4. If full, evict from back (least recent) before inserting"
            ),
            "code": (
                "class LRUCache:\n"
                "    def __init__(self, capacity: int):\n"
                "        self.capacity = capacity\n"
                "        self.cache = {}\n"
                "    def get(self, key):\n"
                "        if key in self.cache:\n"
                "            val = self.cache.pop(key)\n"
                "            self.cache[key] = val\n"
                "            return val\n"
                "        return -1\n"
                "    def put(self, key, value):\n"
                "        if key in self.cache:\n"
                "            self.cache.pop(key)\n"
                "        elif len(self.cache) >= self.capacity:\n"
                "            self.cache.pop(next(iter(self.cache)))\n"
                "        self.cache[key] = value"
            ),
            "verify": (
                "cache = LRUCache(2)\n"
                "cache.put(1, 'a')\ncache.put(2, 'b')\n"
                "print(cache.get(1))  # 'a', key 1 is now most recent\n"
                "cache.put(3, 'c')  # evicts key 2 (least recent)\n"
                "print(cache.get(2))  # -1 (evicted)"
            ),
        },
    ]

    texts = []
    for tmpl in tasks:
        parts = []
        parts.append("<|task|> " + tmpl["task"])
        parts.append("<|think|>")
        parts.append(tmpl["think"])
        parts.append("<|code|>")
        parts.append(tmpl["code"])
        parts.append("<|verify|>")
        parts.append(tmpl["verify"])
        texts.append("\n".join(parts))

    while len(texts) < n_samples:
        extra = []
        for tmpl in tasks:
            parts = []
            parts.append("<|task|> " + tmpl["task"])
            parts.append("<|think|>")
            parts.append(tmpl["think"])
            parts.append("<|code|>")
            parts.append(tmpl["code"])
            parts.append("<|verify|>")
            parts.append(tmpl["verify"])
            extra.append("\n".join(parts))
        random.shuffle(extra)
        texts.extend(extra)
    random.shuffle(texts)
    return texts[:n_samples]


def generate_synthetic_data(n_total: int = 50000, rank_id: int = 0) -> list:
    """Generate Phi-1/SemCoder-inspired synthetic training data.

    Stage distribution:
      1. Mental Simulation (exec traces): 30%
      2. Textbook Explanations: 25%
      3. Monologue Debugging: 20%
      4. Novel API Learning: 15%
      5. Agent Trajectories: 10%
    """
    log(f"Generating {n_total} synthetic texts (rank={rank_id}) ...")
    t0 = time.time()
    all_texts = []

    n1 = int(n_total * 0.30)
    n2 = int(n_total * 0.25)
    n3 = int(n_total * 0.20)
    n4 = int(n_total * 0.15)
    n5 = n_total - n1 - n2 - n3 - n4

    log(f"  Stage 1 (Mental Simulation): {n1} target ...")
    t1 = _stage1_mental_simulation(n1, seed=42 + rank_id * 1000)
    all_texts.extend(t1)

    log(f"  Stage 2 (Textbook Explanations): {n2} target ...")
    t2 = _stage2_textbook_explanations(n2, seed=142 + rank_id * 1000)
    all_texts.extend(t2)

    log(f"  Stage 3 (Monologue Debugging): {n3} target ...")
    t3 = _stage3_monologue_debugging(n3, seed=242 + rank_id * 1000)
    all_texts.extend(t3)

    log(f"  Stage 4 (Novel API Learning): {n4} target ...")
    t4 = _stage4_novel_api(n4, seed=342 + rank_id * 1000)
    all_texts.extend(t4)

    log(f"  Stage 5 (Agent Trajectories): {n5} target ...")
    t5 = _stage5_agent_trajectories(n5, seed=442 + rank_id * 1000)
    all_texts.extend(t5)

    random.shuffle(all_texts)
    log(f"Synthetic generation done in {time.time() - t0:.1f}s — "
        f"{len(all_texts)} texts "
        f"(s1={len(t1)}, s2={len(t2)}, s3={len(t3)}, "
        f"s4={len(t4)}, s5={len(t5)})")
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
        synth_texts = generate_synthetic_data(
            target_synth, rank_id=int(
                os.environ.get("RANK_ID", "0")))
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
