"""train.py — SFM-enhanced Qwen2.5-Coder-1.5B training on 4x Ascend 910.

Single-file, self-contained training script for OpenI.  Runs unattended for up to
2 hours and produces results with ZERO human interaction.  Only MindSpore 2.2
(pre-installed) and Python stdlib are used — no pip installs.

Architecture
-----------
* Qwen2.5-Coder-1.5B backbone (frozen, ~1.5B params)
* LoRA rank=32 alpha=64 on all attention + FFN projections
* SFM Slot Banks (8 slots) at layers 6, 13, 20, 27
* DATA_PARALLEL across 4 NPUs (fallback to single-device)
* Gradient checkpointing (recompute) on every transformer layer
* SOMAS memory optimisation (memory_optimize_level=O1)
"""

import os
import sys
import struct
import json
import time
import math
import gc
import glob
import warnings

# ── Bootstrap diagnostic (before anything else can fail) ──────────────────
try:
    _boot_ts = time.strftime("%H:%M:%S")
    _boot_pid = os.getpid()
    _boot_rank = os.environ.get("RANK_ID", "?")
    sys.stderr.write(f"[BOOT {_boot_ts}] pid={_boot_pid} rank={_boot_rank} started\n")
    sys.stderr.flush()
    # Write to /cache/output directly (bypasses all variable logic)
    os.makedirs("/cache/output", exist_ok=True)
    with open("/cache/output/boot.log", "a") as _bf:
        _bf.write(f"[{_boot_ts}] pid={_boot_pid} rank={_boot_rank} script_started\n")
except Exception:
    pass
import subprocess

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Environment variables (BEFORE any MindSpore import)
# ═══════════════════════════════════════════════════════════════════════════

os.environ.update({
    "MS_COMPILER_CACHE_ENABLE": "1",
    "MS_COMPILER_CACHE_PATH": "/home/ma-user/work/graph_cache",
    "MS_BUILD_PROCESS_NUM": "24",
    "TASK_QUEUE_ENABLE": "2",
    "CPU_AFFINITY_CONF": "1",
    "ASCEND_GLOBAL_LOG_LEVEL": "3",      # errors only
    "GLOG_v": "2",                         # warnings + errors
    "HCCL_CONNECT_TIMEOUT": "1800",
    "MS_COMPILER_OP_LEVEL": "0",
})

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1b — c2net path discovery (OpenI platform)
# ═══════════════════════════════════════════════════════════════════════════

# Check if parent passed paths via env vars (from subprocess launcher)
_PARENT_PASSED = os.environ.get("SFM_OUTPUT_PATH") is not None
HAS_C2NET = False
CODE_PATH = "/home/ma-user/work/code"
DATASET_PATH = "/home/ma-user/work/dataset"
PRETRAIN_MODEL_PATH = "/home/ma-user/work/pretrainmodel"
OUTPUT_PATH = "/home/ma-user/work/output"

if _PARENT_PASSED:
    # Paths already discovered by parent process
    OUTPUT_PATH = os.environ["SFM_OUTPUT_PATH"]
    DATASET_PATH = os.environ["SFM_DATASET_PATH"]
    PRETRAIN_MODEL_PATH = os.environ["SFM_PRETRAIN_PATH"]
    CODE_PATH = os.environ.get("SFM_CODE_PATH", CODE_PATH)
else:
    # Try c2net
    try:
        from c2net.context import prepare, upload_output  # noqa: F401
        _ctx = prepare()
        CODE_PATH = _ctx.code_path
        DATASET_PATH = _ctx.dataset_path
        PRETRAIN_MODEL_PATH = _ctx.pretrain_model_path
        OUTPUT_PATH = _ctx.output_path
        HAS_C2NET = True
        print(f"c2net initialised: code={CODE_PATH}, "
              f"dataset={DATASET_PATH}, "
              f"pretrain={PRETRAIN_MODEL_PATH}, "
              f"output={OUTPUT_PATH}", flush=True)
    except Exception:
        print("c2net not available — using default paths", flush=True)

# Fallback: grampus sets LOCAL_* env vars if c2net didn't provide paths
if not os.path.isdir(CODE_PATH) and os.environ.get("LOCAL_CODE_PATH"):
    CODE_PATH = os.environ["LOCAL_CODE_PATH"]
if not os.path.isdir(DATASET_PATH) and os.environ.get("LOCAL_DATASET_PATH"):
    DATASET_PATH = os.environ["LOCAL_DATASET_PATH"]
if not os.path.isdir(PRETRAIN_MODEL_PATH) and os.environ.get("LOCAL_PRETRAIN_MODEL_PATH"):
    PRETRAIN_MODEL_PATH = os.environ["LOCAL_PRETRAIN_MODEL_PATH"]
if not OUTPUT_PATH or not os.path.isdir(OUTPUT_PATH):
    OUTPUT_PATH = os.environ.get("LOCAL_OUTPUT_PATH", "/cache/output")

# Derive working dirs from discovered paths
CKPT_DIR = os.path.join(OUTPUT_PATH, "checkpoints")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Constants (Qwen2.5-Coder-1.5B config.json values)
# ═══════════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 151936
HIDDEN_SIZE = 1536
NUM_LAYERS = 28
NUM_HEADS = 12
NUM_KV_HEADS = 2
INTERMEDIATE_SIZE = 8960
HEAD_DIM = 128                 # HIDDEN_SIZE // NUM_HEADS
NUM_GROUPS = 6                 # NUM_HEADS // NUM_KV_HEADS
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000.0
MAX_SEQ_LEN = 2048
MAX_POSITION = 32768
TIE_WORD_EMBEDDINGS = True

# LoRA
LORA_RANK = 32
LORA_ALPHA = 64

# SFM
SFM_NUM_SLOTS = 8
SFM_LAYERS = {6, 13, 20, 27}  # 0-indexed

# Training
BATCH_SIZE = 16                # per-device, fallback to 8
SEQ_LEN = 2048
LEARNING_RATE = 2e-4
MIN_LR = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TIME_LIMIT = 7000              # seconds (< 2 h with buffer)

# Infrastructure
RANK_SIZE = 4

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Logging
# ═══════════════════════════════════════════════════════════════════════════

_log_fh = None                  # per-worker file handle

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

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Multi-NPU launcher (parent process, no MindSpore)
# ═══════════════════════════════════════════════════════════════════════════

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
        # Pass discovered paths to children so they skip c2net re-init
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

    # Collect results from rank 0
    results_path = os.path.join(OUTPUT_PATH, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        log("=" * 60)
        log("RESULTS SUMMARY")
        log("=" * 60)
        for k, v in results.items():
            log(f"  {k}: {v}")

    # Upload via c2net
    if HAS_C2NET:
        try:
            upload_output()
            log("upload_output() completed successfully")
        except Exception as e:
            log(f"upload_output() failed: {e}")


# ── Fork gate ──────────────────────────────────────────────────────────────
if is_main_process():
    launch_distributed()
    sys.exit(0)

# Worker process — print early diagnostic
print(f"[worker] RANK_ID={os.environ.get('RANK_ID')}, "
      f"OUTPUT_PATH={OUTPUT_PATH}, PID={os.getpid()}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Safetensors reader (pure Python, no pip package)
# ═══════════════════════════════════════════════════════════════════════════

def load_safetensors(filepath: str) -> dict:
    """Read a .safetensors file and return {name: np.ndarray}."""
    import numpy as np
    tensors = {}
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        data_base = 8 + header_size
        for name, meta in header.items():
            # Skip non-tensor metadata (e.g. "__metadata__")
            if not isinstance(meta, dict) or "dtype" not in meta:
                continue
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            off0, off1 = meta["data_offsets"]
            f.seek(data_base + off0)
            raw = f.read(off1 - off0)
            if dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).copy().reshape(shape)
            elif dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).copy().reshape(shape)
                arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            tensors[name] = arr
    return tensors

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Weight conversion (HF → MindSpore)
# ═══════════════════════════════════════════════════════════════════════════

_HF_ATT = "self_attn"
_HF_MLP = "mlp"
_HF_NORM1 = "input_layernorm"
_HF_NORM2 = "post_attention_layernorm"

def _hf_to_ms_name(hf_name: str) -> str:
    """Convert a HuggingFace param name to our MindSpore model param name."""
    prefix = "model."
    if hf_name.startswith(prefix):
        hf_name = hf_name[len(prefix):]
    if hf_name == "embed_tokens.weight":
        return "embedding.embedding_table"
    if hf_name == "norm.weight":
        return "norm.weight"
    if hf_name == "lm_head.weight":
        return None
    if not hf_name.startswith("layers."):
        return None
    parts = hf_name.split(".")
    layer_i = parts[1]
    rest = ".".join(parts[2:])
    ms_layer = f"layers.{layer_i}"
    if rest == f"{_HF_ATT}.q_proj.weight":
        return f"{ms_layer}.attention.wq.frozen_weight"
    if rest == f"{_HF_ATT}.q_proj.bias":
        return f"{ms_layer}.attention.wq.frozen_bias"
    if rest == f"{_HF_ATT}.k_proj.weight":
        return f"{ms_layer}.attention.wk.frozen_weight"
    if rest == f"{_HF_ATT}.k_proj.bias":
        return f"{ms_layer}.attention.wk.frozen_bias"
    if rest == f"{_HF_ATT}.v_proj.weight":
        return f"{ms_layer}.attention.wv.frozen_weight"
    if rest == f"{_HF_ATT}.v_proj.bias":
        return f"{ms_layer}.attention.wv.frozen_bias"
    if rest == f"{_HF_ATT}.o_proj.weight":
        return f"{ms_layer}.attention.wo.frozen_weight"
    if rest == f"{_HF_ATT}.o_proj.bias":
        return f"{ms_layer}.attention.wo.frozen_bias"
    if rest == f"{_HF_MLP}.gate_proj.weight":
        return f"{ms_layer}.feed_forward.w1.frozen_weight"
    if rest == f"{_HF_MLP}.up_proj.weight":
        return f"{ms_layer}.feed_forward.w3.frozen_weight"
    if rest == f"{_HF_MLP}.down_proj.weight":
        return f"{ms_layer}.feed_forward.w2.frozen_weight"
    if rest == f"{_HF_NORM1}.weight":
        return f"{ms_layer}.attention_norm.weight"
    if rest == f"{_HF_NORM2}.weight":
        return f"{ms_layer}.ffn_norm.weight"
    return None


def find_model_dir(base_path: str) -> str:
    """Search for .safetensors files under base_path (max depth 3).

    Returns the directory containing the files, or None.
    """
    if not os.path.isdir(base_path):
        return None

    # Check base_path itself
    if glob.glob(os.path.join(base_path, "*.safetensors")):
        return base_path

    # Check immediate subdirectories
    for d in sorted(os.listdir(base_path)):
        sub = os.path.join(base_path, d)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.safetensors")):
            return sub

    # Recursive search (max depth 2 more)
    for root, dirs, _files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        if glob.glob(os.path.join(root, "*.safetensors")):
            return root
    return None


def load_pretrained_weights(model, model_dir: str, rank_id: int) -> bool:
    """Load HF safetensors into model.  Returns True on success."""
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        log(f"WARNING: no .safetensors in {model_dir} — using random init")
        return False

    log(f"Loading {len(st_files)} safetensors file(s) …")
    all_tensors = {}
    for sf in st_files:
        log(f"  Reading {os.path.basename(sf)} …")
        all_tensors.update(load_safetensors(sf))

    param_dict = {}
    skipped = 0
    for hf_name, arr in all_tensors.items():
        ms_name = _hf_to_ms_name(hf_name)
        if ms_name is None:
            skipped += 1
            continue
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        param_dict[ms_name] = arr

    log(f"  Mapped {len(param_dict)} params, skipped {skipped} (tied/unused)")

    ms_param = {}
    not_found = []
    for name, _ in model.parameters_and_names():
        if name in param_dict:
            ms_param[name] = ms.Parameter(
                ms.Tensor(param_dict[name]), name=name
            )
        else:
            if any(k in name for k in ("lora_", "sfm", "gate")):
                continue
            not_found.append(name)

    if not_found:
        log(f"  WARNING: {len(not_found)} model params not found in HF weights "
            f"(first 5: {not_found[:5]})")

    ms.load_param_into_net(model, ms_param)
    log(f"  Weights loaded successfully")
    return True

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Model architecture  (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Zero, Normal, One
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F


class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.weight = ms.Parameter(
            ms.Tensor(np.ones(dim, dtype=np.float32)), name="norm_weight"
        )
        self.eps = Tensor(eps, ms.float32)
        self.rsqrt = ops.Rsqrt()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.add = ops.Add()

    def construct(self, x: Tensor) -> Tensor:
        x_float = self.cast(x, ms.float32)
        variance = self.mean(x_float * x_float, -1)
        x_norm = x_float * self.rsqrt(variance + self.eps)
        return self.cast(x_norm, x.dtype) * self.cast(self.weight, x.dtype)


class RotaryEmbedding(nn.Cell):
    """Pre-computed RoPE cos/sin tables.  Applied in GQAAttention."""

    def __init__(self, head_dim: int = HEAD_DIM, max_seq: int = MAX_SEQ_LEN,
                 theta: float = ROPE_THETA):
        super().__init__()
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2,
                                              dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb).astype(np.float16)
        sin = np.sin(emb).astype(np.float16)
        self.cos_table = Tensor(cos[np.newaxis, np.newaxis, :, :])
        self.sin_table = Tensor(sin[np.newaxis, np.newaxis, :, :])

    def construct(self, x: Tensor) -> Tensor:
        cos = self.cos_table
        sin = self.sin_table
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = ops.concat([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class LoRALinear(nn.Cell):
    """Linear layer with frozen base weight and trainable LoRA."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
                 has_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.scaling = Tensor(alpha / rank, ms.float16)
        self.frozen_weight = ms.Parameter(
            ms.Tensor(np.random.randn(out_features, in_features).astype(
                np.float16) * 0.02),
            name="frozen_weight", requires_grad=False,
        )
        self.frozen_bias = None
        if has_bias:
            self.frozen_bias = ms.Parameter(
                ms.Tensor(np.zeros(out_features, dtype=np.float16)),
                name="frozen_bias", requires_grad=False,
            )
        self.lora_A = ms.Parameter(
            ms.Tensor(np.random.randn(rank, in_features).astype(np.float16)
                      * (1.0 / math.sqrt(in_features))),
            name="lora_A",
        )
        self.lora_B = ms.Parameter(
            ms.Tensor(np.zeros((out_features, rank), dtype=np.float16)),
            name="lora_B",
        )
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, x: Tensor) -> Tensor:
        shape = x.shape
        x2 = x.reshape((-1, self.in_features))
        base_out = self.matmul(x2, self.frozen_weight)
        if self.has_bias:
            base_out = base_out + self.frozen_bias
        hidden = self.matmul(x2, self.lora_A)
        lora_out = self.matmul(hidden, self.lora_B) * self.scaling
        return (base_out + lora_out).reshape(shape[:-1] + (self.out_features,))


class GQAAttention(nn.Cell):
    """Multi-head attention with GQA (12 query heads, 2 KV heads)."""

    def __init__(self):
        super().__init__()
        self.wq = LoRALinear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, has_bias=True)
        self.wk = LoRALinear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, has_bias=True)
        self.wv = LoRALinear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, has_bias=True)
        self.wo = LoRALinear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, has_bias=True)
        self.rope = RotaryEmbedding(HEAD_DIM, MAX_SEQ_LEN)
        self.tile = ops.Tile()
        self.reshape = ops.Reshape()
        self.bmm_t = ops.BatchMatMul(transpose_b=True)
        self.bmm = ops.BatchMatMul()
        self.softmax = ops.Softmax(axis=-1)
        self.scale = Tensor(HEAD_DIM ** -0.5, ms.float16)
        mask_np = np.triu(
            np.full((MAX_SEQ_LEN, MAX_SEQ_LEN), -1e4, dtype=np.float16), k=1
        )
        self.causal_mask = Tensor(mask_np[np.newaxis, np.newaxis, :, :])

    def construct(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        S = x.shape[1]
        q = self.wq(x).reshape(B, S, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.tile(k, (1, NUM_GROUPS, 1, 1))
        v = self.tile(v, (1, NUM_GROUPS, 1, 1))
        q = self.rope(q)
        k = self.rope(k)
        scores = self.bmm_t(q, k) * self.scale
        scores = scores + self.causal_mask[:, :, :S, :S]
        attn = self.softmax(scores)
        out = self.bmm(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, NUM_HEADS * HEAD_DIM)
        return self.wo(out)


class SwiGLUFFN(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w1 = LoRALinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, has_bias=False)
        self.w3 = LoRALinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, has_bias=False)
        self.w2 = LoRALinear(INTERMEDIATE_SIZE, HIDDEN_SIZE, has_bias=False)
        self.silu = nn.SiLU()
        self.mul = ops.Mul()

    def construct(self, x: Tensor) -> Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class SFMSlotBank(nn.Cell):
    """State-Flow Machine slot bank with cross-attention + gated summary."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_slots: int = SFM_NUM_SLOTS):
        super().__init__()
        self.num_slots = num_slots
        self.H = hidden_size
        self.slots = ms.Parameter(
            ms.Tensor(np.random.randn(num_slots, hidden_size).astype(
                np.float16) * 0.02),
            name="slot_vectors",
        )
        self.q_proj = nn.Dense(hidden_size, hidden_size)
        self.k_proj = nn.Dense(hidden_size, hidden_size)
        self.v_proj = nn.Dense(hidden_size, hidden_size)
        self.output_proj = nn.Dense(hidden_size, hidden_size)
        self.gate = ms.Parameter(ms.Tensor([0.0], dtype=ms.float32),
                                  name="sfm_gate")
        self.sqrt_d = Tensor(hidden_size ** -0.5, ms.float16)
        self.bmm_t = ops.BatchMatMul(transpose_b=True)
        self.bmm = ops.BatchMatMul()
        self.softmax = ops.Softmax(axis=-1)
        self.sigmoid = ops.Sigmoid()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.tile = ops.Tile()

    def construct(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        sl = self.tile(self.slots.reshape(1, self.num_slots, self.H), (B, 1, 1))
        Q = self.q_proj(sl)        # (B, num_slots, H)
        K = self.k_proj(x)         # (B, S, H)
        V = self.v_proj(x)         # (B, S, H)
        attn = self.softmax(self.bmm_t(Q, K) * self.sqrt_d)  # (B, num_slots, S)
        slot_out = self.bmm(attn, V)                        # (B, num_slots, H)
        summary = self.mean(slot_out, 1)                     # (B, 1, H)
        return x + self.sigmoid(self.gate.astype(x.dtype)) * self.output_proj(summary)


class TransformerBlock(nn.Cell):
    def __init__(self, layer_id: int):
        super().__init__()
        self.attention_norm = RMSNorm(HIDDEN_SIZE)
        self.attention = GQAAttention()
        self.ffn_norm = RMSNorm(HIDDEN_SIZE)
        self.feed_forward = SwiGLUFFN()
        if layer_id in SFM_LAYERS:
            self.sfm_bank = SFMSlotBank()
        else:
            self.sfm_bank = None

    def construct(self, x: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x))
        h = h + self.feed_forward(self.ffn_norm(h))
        if self.sfm_bank is not None:
            h = self.sfm_bank(h)
        return h


class Thinker1Model(nn.Cell):
    """Qwen2.5-Coder-1.5B backbone with LoRA + SFM adapters."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.CellList([
            TransformerBlock(i) for i in range(NUM_LAYERS)
        ])
        self.norm = RMSNorm(HIDDEN_SIZE)
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, input_ids: Tensor) -> Tensor:
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h2 = h.reshape((-1, HIDDEN_SIZE))
        logits = self.matmul(h2, self.embedding.embedding_table)
        return logits.reshape(h.shape[0], h.shape[1], VOCAB_SIZE)


class ForwardLossCell(nn.Cell):
    def __init__(self, model: Thinker1Model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def construct(self, input_ids: Tensor) -> Tensor:
        logits = self.model(input_ids)
        logits_t = logits[:, :-1, :]
        labels = input_ids[:, 1:].reshape((-1,))
        loss = self.loss_fn(logits_t.reshape((-1, VOCAB_SIZE)), labels)
        return loss


class TrainStep(nn.Cell):
    def __init__(self, forward_loss: ForwardLossCell, optimizer,
                 max_grad_norm: float = MAX_GRAD_NORM):
        super().__init__()
        self.forward_loss = forward_loss
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = Tensor([1.0], ms.float32)
        self.depend = ops.Depend()
        self.clip_grad = nn.ClipByGlobalNorm(max_norm=max_grad_norm)

    def construct(self, input_ids: Tensor) -> Tensor:
        loss = self.forward_loss(input_ids)
        grads = self.grad_op(self.forward_loss, self.weights)(input_ids, self.sens)
        clipped, _ = self.clip_grad(grads)
        status = self.optimizer(clipped)
        return self.depend(loss, status)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Dataset loading (OpenThoughts-114k or synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text(row: dict) -> str:
    """Extract text content from a dataset row (various schema formats)."""
    # Chat / conversations format
    for key in ("messages", "conversations"):
        if key in row:
            parts = row[key]
            if isinstance(parts, list):
                return "\n".join(
                    m.get("content", m.get("text", "")) for m in parts
                    if isinstance(m, dict)
                )
    # Field-based formats
    for pair in [("output", "input"), ("response", "instruction"),
                  ("solution", "problem"), ("answer", "question"),
                  ("content", "text")]:
        a, b = pair
        if a in row and b in row:
            return str(row[b]) + "\n" + str(row[a])
        if a in row:
            return str(row[a])
    return str(row)


def _find_data_files(base: str) -> list:
    """Recursively find readable data files under base."""
    found = []
    for ext in ("*.jsonl", "*.json", "*.parquet", "*.csv", "*.txt", "*.arrow"):
        found.extend(glob.glob(os.path.join(base, "**", ext), recursive=True))
    return sorted(found)


def _try_load_byte_tokenizer(pretrain_path: str):
    """Try to build a vocab mapping from tokenizer.json."""
    tok_path = None
    # Direct path first
    for cand in [
        os.path.join(pretrain_path, "tokenizer.json"),
    ]:
        if os.path.isfile(cand):
            tok_path = cand
            break
    # Recursive search (tokenizer may be in a subdirectory)
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
        vocab = tok_data.get("model", {}).get("vocab", {})
        if not vocab:
            return None
        token_to_id = {}
        for token, idx in vocab.items():
            token_to_id[token] = idx
        log(f"  Loaded BPE tokenizer: {len(token_to_id)} tokens from "
            f"{tok_path}")
        return token_to_id
    except Exception as e:
        log(f"  Failed to parse tokenizer.json: {e}")
        return None


def _tokenize_texts(texts: list, token_to_id: dict, seq_len: int) -> np.ndarray:
    """Convert text strings to token-id chunks of fixed seq_len using BPE vocab.
    Falls back to byte-level encoding if vocab is unavailable.
    Returns (num_chunks, seq_len) int32 array.
    """
    byte_fallback = (token_to_id is None or len(token_to_id) < 100)

    if not byte_fallback:
        # BPE tokenisation — try whole-word lookup, then character fallback
        all_ids = []
        unk_id = token_to_id.get("<unk>", 0)
        for text in texts:
            ids = [unk_id] * (2)  # BOS-like padding
            words = text.split()
            for word in words:
                tid = token_to_id.get(word)
                if tid is not None:
                    ids.append(tid)
                else:
                    # Fallback: character-level greedy match
                    i = 0
                    while i < len(word):
                        matched = False
                        # Try longest match first
                        for end in range(min(len(word), i + 20), i, -1):
                            sub = word[i:end]
                            if sub in token_to_id:
                                ids.append(token_to_id[sub])
                                i = end
                                matched = True
                                break
                        if not matched:
                            ids.append(unk_id)
                            i += 1
            ids.append(unk_id)  # EOS-like
            all_ids.extend(ids)
        all_ids = np.array(all_ids, dtype=np.int64)
    else:
        # Byte-level fallback
        log("  Using byte-level tokenizer (no BPE vocab found)")
        all_ids = np.zeros(0, dtype=np.int32)
        for text in texts:
            encoded = text.encode("utf-8", errors="replace")
            all_ids = np.concatenate([all_ids, np.frombuffer(encoded,
                                                             dtype=np.uint8).astype(np.int32)])
        # Add BOS/EOS sentinel bytes (256 = outside ASCII range)
        all_ids = np.concatenate([
            np.full(2, 256, dtype=np.int64),  # BOS pair
            all_ids,
            np.full(1, 256, dtype=np.int64),  # EOS
        ])

    total = len(all_ids)
    # Trim to exact multiple of seq_len
    usable = (total // seq_len) * seq_len
    if usable < seq_len:
        log("  WARNING: dataset too small for even one chunk, using random fallback")
        return None
    trimmed = all_ids[:usable]
    chunks = trimmed.reshape(-1, seq_len)
    log(f"  {len(chunks)} chunks of {seq_len} tokens from "
        f"{total:,} tokens "
        f"(byte_fallback={byte_fallback})")
    return chunks.astype(np.int32)


def load_dataset(dataset_path: str, seq_len: int,
               pretrain_path: str):
    """Load and tokenise the OpenThoughts-114k dataset.

    Returns (num_chunks, seq_len) int32 array, or None on failure.
    """
    files = _find_data_files(dataset_path)
    log(f"  Found {len(files)} data file(s) in {dataset_path}")
    if not files:
        return None

    # Read all rows
    texts = []
    for fp in files:
        log(f"  Reading {os.path.basename(fp)} …")
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
                log(f"    (parquet, {len(texts)} rows extracted)")
            elif fp.endswith(".csv"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            texts.append(_extract_text(
                                json.loads("{\"text\": " + json.dumps(line) + "}")))
            elif fp.endswith(".txt"):
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            else:
                log(f"    Skipping unknown format: {os.path.basename(fp)}")
        except Exception as e:
            log(f"    Error reading {fp}: {e}")

    if not texts:
        log("  WARNING: no text content extracted from any data file")
        return None

    log(f"  Extracted {len(texts)} text samples")

    # Deduplicate and filter empty
    texts = list(set(t for t in texts if len(t.strip()) > 10))
    log(f"  After dedup: {len(texts)} unique samples")

    # Try BPE tokenizer
    token_to_id = _try_load_byte_tokenizer(pretrain_path)

    return _tokenize_texts(texts, token_to_id, seq_len)


# Synthetic fallback dataset
class RandomTokenDataset:
    """Synthetic random-token dataset for throughput measurement."""

    def __init__(self, num_samples: int, seq_len: int = SEQ_LEN):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return np.random.randint(0, VOCAB_SIZE, (self.seq_len,)).astype(np.int32)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Cell):
    total = sum(p.size for p in model.get_parameters())
    trainable = sum(p.size for p in model.trainable_params())
    return total, trainable


def main():
    t0_global = time.time()

    # ── Rank / device ────────────────────────────────────────────────────
    rank_id = int(os.environ.get("RANK_ID", "0"))
    rank_size = int(os.environ.get("RANK_SIZE", "1"))
    device_id = int(os.environ.get("DEVICE_ID", "0"))

    setup_logging(rank_id)

    log("=" * 60)
    log(f"Thinker-1 training | rank={rank_id} device={device_id} "
        f"world_size={rank_size}")
    log(f"Python {sys.version}")
    log(f"CWD: {os.getcwd()}")
    log("")

    # ── Print discovered paths ──────────────────────────────────────────────
    log(f"CODE_PATH:     {CODE_PATH}")
    log(f"DATASET_PATH:  {DATASET_PATH}")
    log(f"PRETRAIN_PATH: {PRETRAIN_MODEL_PATH}")
    log(f"OUTPUT_PATH:   {OUTPUT_PATH}")
    log("")

    # List dataset directory contents
    if os.path.isdir(DATASET_PATH):
        entries = sorted(os.listdir(DATASET_PATH))[:20]
        log(f"DATASET dir contents: {entries}")
    else:
        log("DATASET_PATH does not exist")

    # Find pretrained model
    model_dir = find_model_dir(PRETRAIN_MODEL_PATH)
    if model_dir:
        log(f"Model weights found at: {model_dir}")
    else:
        log(f"WARNING: no .safetensors found under {PRETRAIN_MODEL_PATH}")

    # ── MindSpore context ────────────────────────────────────────────────
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        memory_optimize_level="O1",
    )

    # ── Data-parallel init (best-effort, fallback to single) ───────────
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

    # ── Build model ──────────────────────────────────────────────────────
    log("Building Thinker-1 model …")
    model = Thinker1Model()

    for i, layer in enumerate(model.layers):
        layer.recompute()

    total_p, trainable_p = count_params(model)
    log(f"Total params: {total_p:,}  |  Trainable: {trainable_p:,}  "
        f"({100*trainable_p/total_p:.2f}%)")

    # ── Freeze backbone, keep LoRA + SFM trainable ───────────────────────
    # Freeze ALL params first, then selectively unfreeze adapters
    for p in model.get_parameters():
        p.requires_grad = False
    for name, p in model.parameters_and_names():
        if "lora_" in name or "sfm_bank" in name:
            p.requires_grad = True
    trainable_p2 = sum(p.size for p in model.trainable_params())
    log(f"Trainable after freeze: {trainable_p2:,}")

    # ── Load pretrained weights ────────────────────────────────────────────────
    # All workers load from safetensors in parallel (avoids ckpt race condition
    # on network filesystem — save/load dance caused DecodeError).
    if model_dir:
        load_pretrained_weights(model, model_dir, rank_id)
    else:
        log("WARNING: no pretrained weights found — using random init")

    # ── Load dataset ─────────────────────────────────────────────────────
    log("Loading dataset …")
    data_chunks = load_dataset(DATASET_PATH, SEQ_LEN, PRETRAIN_MODEL_PATH)
    use_real_data = data_chunks is not None

    if use_real_data:
        log(f"Using REAL dataset: {data_chunks.shape[0]} chunks, "
            f"{data_chunks.shape[1]} tok/chunk")
    else:
        log("WARNING: no dataset files found — using synthetic random tokens")

    # ── Optimiser ────────────────────────────────────────────────────────
    tokens_per_step_per_device = BATCH_SIZE * SEQ_LEN
    target_tokens_per_device = 125_000_000  # 500M / 4
    if use_real_data:
        max_steps = min(
            target_tokens_per_device // tokens_per_step_per_device + 10,
            data_chunks.shape[0] // rank_size + 10,   # don't over-iterate data
        )
    else:
        max_steps = target_tokens_per_device // tokens_per_step_per_device + 10
    actual_lr = LEARNING_RATE
    log(f"Target: {target_tokens_per_device:,} tokens/device, "
        f"{max_steps} steps, B={BATCH_SIZE} S={SEQ_LEN}, "
        f"real_data={use_real_data}")

    train_params = list(model.trainable_params())
    optimizer = nn.AdamWeightDecay(
        train_params,
        learning_rate=Tensor(actual_lr, ms.float32),
        weight_decay=WEIGHT_DECAY,
        beta1=0.9,
        beta2=0.95,
    )

    # ── Training cells (try B=16, fallback to B=8) ──────────────────────
    forward_loss = ForwardLossCell(model)
    train_step = None
    actual_bs = BATCH_SIZE

    for bs in [BATCH_SIZE, BATCH_SIZE // 2]:
        try:
            log(f"Building training graph for B={bs} …")
            ts = TrainStep(forward_loss, optimizer, MAX_GRAD_NORM)
            ts.set_train()
            dummy = Tensor(
                np.random.randint(0, VOCAB_SIZE, (bs, SEQ_LEN)).astype(np.int32)
            )
            log("  Compiling (first step) …")
            _ = ts(dummy)
            log(f"  B={bs} compilation OK")
            train_step = ts
            actual_bs = bs
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "alloc" in msg:
                log(f"  B={bs} OOM, trying smaller batch …")
                del ts
                gc.collect()
                continue
            log(f"  B={bs} FAILED: {e}")
            raise

    if train_step is None:
        log("FATAL: all batch sizes OOM")
        return

    log(f"Training with B={actual_bs}, S={SEQ_LEN}, "
        f"{actual_bs * SEQ_LEN:,} tok/step/device")

    # ── Training loop ────────────────────────────────────────────────────
    step = 0
    total_tokens = 0
    best_loss = float("inf")
    t_start = time.time()
    tokens_per_step = actual_bs * SEQ_LEN

    log("")
    log("=" * 60)
    log("TRAINING START")
    log("=" * 60)

    while step < max_steps:
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            log(f"Time limit reached ({elapsed:.0f}s > {TIME_LIMIT}s)")
            break

        # Get next batch
        if use_real_data:
            # Distribute chunks across ranks with modular indexing
            indices = [(step * actual_bs + rank_id * actual_bs + j) % data_chunks.shape[0]
                       for j in range(actual_bs)]
            batch_np = data_chunks[indices]
        else:
            batch_np = np.random.randint(0, VOCAB_SIZE,
                                         (actual_bs, SEQ_LEN)).astype(np.int32)
        batch_tensor = Tensor(batch_np)

        try:
            loss_val = train_step(batch_tensor)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "alloc" in msg:
                log(f"OOM at step {step}! Stopping training.")
                break
            raise

        loss_float = float(loss_val.asnumpy())
        total_tokens += tokens_per_step * rank_size
        if loss_float < best_loss:
            best_loss = loss_float

        step += 1

        if step % 50 == 0 or step <= 3:
            dt = time.time() - t_start
            tps_dev = tokens_per_step / dt * step if dt > 0 else 0
            tps_total = tps_dev * rank_size
            log(f"Step {step:>5d} | loss={loss_float:.4f} | "
                f"best={best_loss:.4f} | "
                f"tok/s/dev={tps_dev:.0f} | "
                f"tok/s/total={tps_total:.0f} | "
                f"lr={actual_lr:.2e} | "
                f"elapsed={dt:.0f}s | "
                f"tokens={total_tokens:,}")

        if rank_id == 0 and step % 1000 == 0 and step > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"step_{step}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            log(f"Checkpoint saved: {ckpt_path}")

    # ── Final checkpoint + results ───────────────────────────────────────
    elapsed_total = time.time() - t_start
    tps_dev_avg = (total_tokens / rank_size) / elapsed_total if elapsed_total > 0 else 0
    tps_total_avg = total_tokens / elapsed_total if elapsed_total > 0 else 0

    if rank_id == 0:
        ms.save_checkpoint(model, os.path.join(CKPT_DIR, "final.ckpt"))
        log(f"Final checkpoint saved")

    results = {
        "model_name": "Thinker-1 (Qwen2.5-Coder-1.5B + LoRA + SFM)",
        "total_params": total_p,
        "trainable_params": trainable_p2,
        "batch_size": actual_bs,
        "seq_len": SEQ_LEN,
        "num_devices": rank_size,
        "total_steps": step,
        "total_tokens": total_tokens,
        "total_time_s": round(elapsed_total, 1),
        "tokens_per_sec_per_device": round(tps_dev_avg, 1),
        "tokens_per_sec_total": round(tps_total_avg, 1),
        "final_loss": round(loss_float, 4) if 'loss_float' in dir() else None,
        "best_loss": round(best_loss, 4),
        "data_parallel": use_dp,
        "used_real_data": use_real_data,
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

    # Upload via c2net
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
        # Write to BOTH /cache/output (hardcoded) and OUTPUT_PATH
        for _err_dir in ["/cache/output", OUTPUT_PATH]:
            try:
                os.makedirs(_err_dir, exist_ok=True)
                with open(os.path.join(_err_dir, "error.log"), "w") as f:
                    f.write(msg)
            except Exception:
                pass
        # Try c2net upload so error.log is downloadable
        try:
            if HAS_C2NET:
                from c2net.context import upload_output
                upload_output()
        except Exception:
            pass
        raise
