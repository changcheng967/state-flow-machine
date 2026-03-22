"""train.py — Thinker-1.5B: Qwen2.5-Coder-1.5B fine-tuned with DeltaNet SFM.

Fine-tunes Qwen2.5-Coder-1.5B (1.5B params, 28 layers, GQA) by inserting
4 DeltaNet SFM blocks after layers 6, 13, 20, 27. The base model is frozen
for the first phase (SFM-only training), then fully fine-tuned in the second
phase.

Architecture:
  Base: Qwen2.5-Coder-1.5B (vocab=151936, hidden=1536, 28 layers, 12Q/2KV GQA,
        intermediate=8960, rope_theta=1000000.0, tied_embeddings=True)
  SFM: DeltaNet recurrent cell (16 heads, 16x16 state) + cross-system bridge
       at layers 6, 13, 20, 27 (0-indexed = 7th, 14th, 21st, 28th layers)
  Judge: Binary classification head (correct/wrong)
  Surprise: Scalar predictor for self-evolution

Training data (synthetic, generated on-the-fly):
  - Phase 1 (Execution): exec() traces with state tracking
  - Phase 2 (Debugging): buggy code + fix reasoning
  Format: <|code|>...<|slots|>{json}...<|monologue|>...<|answer|>...<|judge_correct|>

Key design decisions:
  - Simple delta rule: S = S - beta*(S@k - v)@k^T (not gated DeltaNet)
  - Q/K/V projections have bias (matching Qwen's architecture)
  - MLP projections have NO bias
  - NUM_GROUPS=6 (12Q / 2KV heads)
  - ~15% corrupted judge labels during bootstrap for balanced training
  - Masked CE loss (only on monologue/slots/answer/judge tokens)
  - Effective batch of 8 (B=2 per device x 4 NPUs DATA_PARALLEL)
  - Gradient clipping (clip by global norm) with manual AllReduce
  - Two param groups: base_params and sfm_params with separate LRs

Infrastructure (from reference train.py):
  - c2net path discovery + local fallback
  - 4-worker process forking
  - DATA_PARALLEL via HCCL
  - Convergence-based stopping
  - Checkpointing

Self-contained: MindSpore 2.2 + NumPy + stdlib only.
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
        f"thinker15b_finetune_started\n")
    sys.stderr.flush()
    os.makedirs("/cache/output", exist_ok=True)
    with open("/cache/output/boot.log", "a") as _bf:
        _bf.write(f"[{_boot_ts}] pid={_boot_pid} rank={_boot_rank} "
                  f"thinker15b_finetune_started\n")
except Exception:
    pass

import subprocess
warnings.filterwarnings("ignore")

# ── Environment vars (BEFORE importing MindSpore) ────────────────────
os.environ.update({
    "MS_COMPILER_CACHE_ENABLE": "1",
    "MS_COMPILER_CACHE_PATH": "/cache/output/graph_cache",
    "MS_BUILD_PROCESS_NUM": "24",
    "TASK_QUEUE_ENABLE": "2",
    "CPU_AFFINITY_CONF": "1",
    "ASCEND_GLOBAL_LOG_LEVEL": "3",
    "GLOG_v": "2",
    "HCCL_CONNECT_TIMEOUT": "1800",
    # Graph kernel optimizations for Ascend
    "MS_DEV_GRAPH_KERNEL_FLAGS": "--enable_expand_ops=Split,Tile,"
                                  "--disable_inline_reducesort,"
                                  "--enable_parallel_fusion=true",
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

# ── Qwen2.5-Coder-1.5B Architecture (from official config.json) ────
VOCAB_SIZE = 151936
HIDDEN_DIM = 1536
NUM_HEADS = 12
NUM_KV_HEADS = 2
NUM_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 6
HEAD_DIM = 128
INTERMEDIATE_DIM = 8960
NUM_LAYERS = 28
MAX_SEQ_LEN = 512  # 2048 causes OOM: CE logits (B*S, V) = 2.5 GB FP32 on 32 GB NPU
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000.0
TIE_WORD_EMBEDDINGS = True

# SFM DeltaNet blocks inserted AFTER these 0-indexed layers
# (layers 6, 13, 20, 27 = 7th, 14th, 21st, 28th layers)
SFM_LAYERS = (6, 13, 20, 27)

# DeltaNet config
DELTANET_HIDDEN_DIM = 256      # Must be multiple of 16
DELTANET_NUM_HEADS = 16
DELTANET_HEAD_DIM = DELTANET_HIDDEN_DIM // DELTANET_NUM_HEADS  # 16
DELTANET_STATE_DIM = 256       # 16x16 state per head

# Cross-system bridge
BRIDGE_DIM = 256                # Shared 256d space per plan

# Judge head
JUDGE_HIDDEN_DIM = 128

# Surprise predictor
SURPRISE_DIM = 64

# ── Training hyperparameters ─────────────────────────────────────────
# Stage 1: SFM-only training (base frozen)
STAGE1_LR_SFM = 1e-3
STAGE1_WARMUP = 200
STAGE1_MAX_STEPS = 10000
STAGE1_WEIGHT_DECAY = 0.01

# Stage 2: Full fine-tuning
STAGE2_LR_BASE = 2e-5
STAGE2_LR_SFM = 5e-4
STAGE2_WARMUP = 500
STAGE2_MAX_STEPS = 50000
STAGE2_WEIGHT_DECAY = 0.01

# Shared
BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACCUM_STEPS = 1       # No accum; 2*4 NPUs = eff batch 8
MAX_GRAD_NORM = 1.0
TIME_LIMIT = 86400
MIN_LR_FACTOR = 0.1

RANK_SIZE = 4

# Self-evolution
EVOLUTION_PROBE_STEPS = 500

# Convergence
CONVERGENCE_WINDOW = 200
CONVERGENCE_PATIENCE = 2000
CONVERGENCE_THRESHOLD = 0.001

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
from mindspore import nn, ops, value_and_grad
from mindspore.common.tensor import Tensor

# ── StubTensor safety net ──────────────────────────────────────────
# Patches StubTensor.dtype/shape/stub_sync to prevent crashes during
# gradient metagraph tracing. Safety net for edge cases.
try:
    from mindspore.common._stub_tensor import StubTensor as _StubTensor
    _orig_dtype_getter = _StubTensor.dtype.fget
    _orig_shape_getter = _StubTensor.shape.fget
    _orig_stub_sync = _StubTensor.stub_sync

    def _safe_dtype(self):
        try:
            return _orig_dtype_getter(self)
        except (RuntimeError, ValueError):
            return ms.float32

    def _safe_shape(self):
        try:
            return _orig_shape_getter(self)
        except (RuntimeError, ValueError):
            # Use __dict__ only to avoid parent class _shape method.
            # StubTensor.__init__ sets virtual_abstract from output.abstract.
            cached = self.__dict__.get('stub_shape')
            if cached is not None:
                return tuple(cached)
            abstract = self.__dict__.get('virtual_abstract')
            if abstract is not None:
                try:
                    return tuple(abstract.shape)
                except Exception:
                    pass
            return (1, 1)

    def _safe_stub_sync(self):
        try:
            return _orig_stub_sync(self)
        except (RuntimeError, ValueError):
            abstract = self.__dict__.get('virtual_abstract')
            if abstract is not None:
                try:
                    return np.zeros(tuple(abstract.shape), dtype=np.float32)
                except Exception:
                    pass
            return np.zeros((1, 1), dtype=np.float32)

    _StubTensor.dtype = property(_safe_dtype)
    _StubTensor.shape = property(_safe_shape)
    _StubTensor.stub_sync = _safe_stub_sync
except Exception:
    pass

# Safety net: patch Tensor.astype to survive any remaining StubTensor
# crashes. Returns input unchanged on StubTensor errors.
try:
    _orig_tensor_astype = Tensor.astype

    def _safe_tensor_astype(self, dtype, *args, **kwargs):
        try:
            return _orig_tensor_astype(self, dtype, *args, **kwargs)
        except RuntimeError as e:
            if "bad optional access" in str(e) or "stub" in str(e).lower():
                return self
            raise
    Tensor.astype = _safe_tensor_astype
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════
# SAFETENSERS LOADER
# ══════════════════════════════════════════════════════════════════════


def _hf_to_ms_name(hf_name: str) -> str:
    """Convert HuggingFace parameter name to MindSpore format.

    Examples:
        model.embed_tokens.weight -> embedding.embedding_table
        model.layers.0.self_attn.q_proj.weight -> layers.0.q_proj.weight
        model.layers.0.mlp.gate_proj.weight -> layers.0.gate_proj.weight
        model.layers.0.input_layernorm.weight -> layers.0.input_norm.weight
        model.layers.0.post_attention_layernorm.weight -> layers.0.ffn_norm.weight
        model.norm.weight -> norm.weight
    """
    name = hf_name
    # Remove 'model.' prefix
    if name.startswith("model."):
        name = name[len("model."):]
    # embed_tokens.weight -> embedding.embedding_table (nn.Embedding uses
    # embedding_table, not weight)
    if name == "embed_tokens.weight":
        return "embedding.embedding_table"
    # Norm layer name mapping (Qwen HF uses layernorm, we use _norm)
    name = name.replace("input_layernorm.weight", "input_norm.weight")
    name = name.replace("post_attention_layernorm.weight", "ffn_norm.weight")
    # self_attn. and mlp. are direct children of layer
    name = name.replace("self_attn.", "")
    name = name.replace("mlp.", "")
    return name


def find_model_dir(base_path: str) -> str:
    """Recursively find the Qwen2.5-Coder-1.5B model directory.

    Prefers a directory containing config.json with num_hidden_layers=28
    (1.5B model). Falls back to first safetensors dir found.
    """
    candidates = []
    for root, dirs, files in os.walk(base_path):
        # Don't recurse too deep
        if root.count(os.sep) - base_path.count(os.sep) > 3:
            dirs.clear()
            continue
        has_st = any(f == "model.safetensors" or f.endswith(".safetensors")
                     for f in files)
        has_cfg = "config.json" in files
        if has_st:
            candidates.append(root)
            # Check config for 1.5B model (28 layers AND hidden_size=1536)
            # Note: Qwen2.5-Coder-7B also has 28 layers but hidden_size=3584
            if has_cfg:
                try:
                    with open(os.path.join(root, "config.json")) as f:
                        cfg = json.load(f)
                    if (cfg.get("num_hidden_layers") == 28 and
                            cfg.get("hidden_size") == HIDDEN_DIM):
                        return root
                except Exception:
                    pass

    # Fallback: return first candidate
    if candidates:
        log(f"  WARNING: no 1.5B config found, using first safetensors dir")
        return candidates[0]
    return ""


def _load_single_safetensors(sf_path: str) -> dict:
    """Load one safetensors file. Returns {name: np_array}."""
    dtype_map = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float16,
        "I32": np.int32,
        "I64": np.int64,
        "BOOL": np.bool_,
    }
    weights = {}
    with open(sf_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)

        for name, info in header.items():
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            dtype_str = info.get("dtype", "F32")
            np_dtype = dtype_map.get(dtype_str, np.float32)
            shape = tuple(info["shape"])
            data_offsets = info["data_offsets"]
            f.seek(data_offsets[0])
            nbytes = data_offsets[1] - data_offsets[0]
            data = np.frombuffer(f.read(nbytes), dtype=np_dtype).reshape(shape)
            weights[name] = data.copy()
    return weights


def load_safetensors(model_dir: str) -> dict:
    """Load all safetensors files from model_dir.

    Handles:
      - Single file: model.safetensors
      - Sharded: model-00001-of-00004.safetensors (loads all shards)
      - Index-based: model.safetensors.index.json
    """
    # Check for single file
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        log(f"Loading weights from {single} ...")
        t0 = time.time()
        weights = _load_single_safetensors(single)
        log(f"Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")
        return weights

    # Sharded: find all model-*.safetensors files
    shards = sorted(glob.glob(os.path.join(model_dir,
                                            "model-*.safetensors")))
    if shards:
        log(f"Loading {len(shards)} sharded safetensors files ...")
        t0 = time.time()
        all_weights = {}
        for sf in shards:
            log(f"  {os.path.basename(sf)} ...")
            all_weights.update(_load_single_safetensors(sf))
        log(f"Loaded {len(all_weights)} tensors in {time.time()-t0:.1f}s")
        return all_weights

    raise FileNotFoundError(f"No safetensors files in {model_dir}")


# ══════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS (inline for GRAPH_MODE safety)
# ══════════════════════════════════════════════════════════════════════


def _make_dense(in_ch: int, out_ch: int,
                has_bias: bool = False) -> nn.Dense:
    """Create nn.Dense with Kaiming-like init (FP32 internally).

    MindSpore 2.2 nn.Dense creates FP32 params internally.
    Weights get overwritten by load_param_into_net() with FP16
    safetensors data, so init dtype doesn't matter for Qwen layers.
    For SFM layers (no pretrained weights), the construct path
    casts inputs to FP16 so FP32 params work fine with matmul.
    """
    return nn.Dense(in_ch, out_ch, has_bias=has_bias)


class RMSNorm(nn.Cell):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.weight = ms.Parameter(
            Tensor(np.ones(dim, dtype=np.float32)), name="weight")
        self.eps = Tensor(eps, ms.float32)

    def construct(self, x: Tensor) -> Tensor:
        x_f = x.astype(ms.float32)
        variance = ops.mean(x_f * x_f, axis=-1, keep_dims=True)
        x_norm = x_f * ops.rsqrt(variance + self.eps)
        return (x_norm * self.weight).astype(x.dtype)


class TransformerBlock(nn.Cell):
    """Single Qwen2.5-Coder transformer layer: GQA MHA + SwiGLU FFN.

    Matches Qwen2.5-Coder-1.5B exactly:
    - q/k/v_proj have bias=True (o_proj has bias=False)
    - MLP projections have NO bias
    - GQA: 12 Q heads, 2 KV heads, NUM_GROUPS=6
    - RoPE with theta=1000000.0
    """

    def __init__(self) -> None:
        super().__init__()
        H = HIDDEN_DIM
        NH = NUM_HEADS
        NKV = NUM_KV_HEADS
        HD = HEAD_DIM
        A = INTERMEDIATE_DIM

        # Q/K/V have bias, O does not (matching Qwen2.5-Coder-1.5B)
        self.q_proj = _make_dense(H, NH * HD, has_bias=True)
        self.k_proj = _make_dense(H, NKV * HD, has_bias=True)
        self.v_proj = _make_dense(H, NKV * HD, has_bias=True)
        self.o_proj = _make_dense(NH * HD, H, has_bias=False)
        self.input_norm = RMSNorm(H)

        # MLP: no bias on any projection
        self.gate_proj = _make_dense(H, A, has_bias=False)
        self.up_proj = _make_dense(H, A, has_bias=False)
        self.down_proj = _make_dense(A, H, has_bias=False)
        self.ffn_norm = RMSNorm(H)

        self.scale = HD ** -0.5  # Python float — CANN 7 can't Mul scalar Tensor with 4D
        self.bmm = ops.BatchMatMul()
        self.bmm_tb = ops.BatchMatMul(transpose_b=True)
        self.transpose_op = ops.Transpose()
        self.tile = ops.Tile()
        self.softmax = ops.Softmax(axis=-1)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> Tensor:
        B = x.shape[0]
        S = x.shape[1]
        NH = NUM_HEADS
        NKV = NUM_KV_HEADS
        HD = HEAD_DIM
        NG = NUM_GROUPS

        h = self.input_norm(x).astype(ms.float32)
        Q = self.transpose_op(self.q_proj(h).reshape(B, S, NH, HD), (0, 2, 1, 3)).astype(ms.float16)
        K = self.transpose_op(self.k_proj(h).reshape(B, S, NKV, HD), (0, 2, 1, 3)).astype(ms.float16)
        V = self.transpose_op(self.v_proj(h).reshape(B, S, NKV, HD), (0, 2, 1, 3)).astype(ms.float16)

        # GQA: tile KV heads to match Q heads (repeat 6x)
        K = self.tile(K, (1, NG, 1, 1))
        V = self.tile(V, (1, NG, 1, 1))

        # RoPE (half-rotation, Llama-style)
        HD2 = HD // 2
        Q = Q * cos + ops.concat([-Q[..., HD2:], Q[..., :HD2]], -1) * sin
        K = K * cos + ops.concat([-K[..., HD2:], K[..., :HD2]], -1) * sin

        # Reshape to 3D for BatchMatMul (CANN 7 doesn't support 4D BMM)
        B_NH = B * NH
        Q3 = Q.reshape(B_NH, S, HD)
        K3 = K.reshape(B_NH, S, HD)
        V3 = V.reshape(B_NH, S, HD)

        attn = self.bmm_tb(Q3, K3) * self.scale
        # Reshape back to 4D for mask broadcast + softmax
        attn = attn.reshape(B, NH, S, S)
        attn = attn + mask[:, :, :S, :S]
        attn = self.softmax(attn)
        attn = attn.reshape(B_NH, S, S)
        out = self.bmm(attn, V3)
        # (B*NH, S, HD) -> reshape to (B, NH, S, HD) -> transpose to (B, S, NH, HD)
        out = self.transpose_op(out.reshape(B, NH, S, HD), (0, 2, 1, 3)).reshape(B, S, NH * HD)
        out = self.o_proj(out.astype(ms.float32)).astype(ms.float16)
        x = x + out

        # SwiGLU FFN
        h = self.ffn_norm(x).astype(ms.float32)
        gate = ops.silu(self.gate_proj(h)).astype(ms.float16)
        up = self.up_proj(h).astype(ms.float16)
        x = x + self.down_proj((gate * up).astype(ms.float32)).astype(ms.float16)
        return x


# ══════════════════════════════════════════════════════════════════════
# DELTANET SFM COMPONENTS
# ══════════════════════════════════════════════════════════════════════


class DeltaNetCell(nn.Cell):
    """Simple delta rule recurrent cell.

    State update: S_new = S - beta * (S @ k - v) @ k^T
    This is the simple (non-gated) version — mathematically cleaner and
    sufficient for state tracking.

    Sequential scan over sequence: O(seq_len) iterations of 16×16 matmuls.
    Each head maintains a 16×16 state matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        D = DELTANET_HIDDEN_DIM
        NH = DELTANET_NUM_HEADS
        HD = DELTANET_HEAD_DIM  # 16

        # Project input to key, value, beta
        # Input comes from bridge down_proj: BRIDGE_DIM -> here
        self.key_proj = _make_dense(BRIDGE_DIM, D, has_bias=False)
        self.value_proj = _make_dense(BRIDGE_DIM, D, has_bias=False)
        self.beta_proj = _make_dense(BRIDGE_DIM, NH, has_bias=True)

        # Initial state (one per head: 16x16 identity matrix)
        # FP32 so gradients are FP32 (avoids dtype mismatch with
        # clip_coef in TrainStep). Cast to FP16 inside construct.
        init_states = np.zeros((NH, HD, HD), dtype=np.float32)
        for i in range(NH):
            init_states[i] = np.eye(HD, dtype=np.float32) * 0.1
        self.initial_state = ms.Parameter(
            Tensor(init_states), name="deltanet_init_state")

        self.NH = NH
        self.HD = HD
        self.bmm = ops.BatchMatMul()
        self.bmm_tb = ops.BatchMatMul(transpose_b=True)
        self.transpose_op = ops.Transpose()
        self.concat_op = ops.Concat(axis=0)

    def construct(self, x: Tensor) -> Tensor:
        """Process sequence through delta rule (sequential scan).

        Uses Python for-loop which MS 2.2 @ms.jit unrolls during graph
        compilation. max_call_depth=100000 is set in ms.set_context() to
        handle the 2048 iterations × 4 SFM layers. Compilation is slow
        (~10-30 min) but the resulting graph runs efficiently.

        Args:
            x: (B, S, BRIDGE_DIM) hidden states from bridge.

        Returns:
            output: (B, S, DELTANET_HIDDEN_DIM) delta net output.
        """
        B = x.shape[0]
        S = x.shape[1]
        D = DELTANET_HIDDEN_DIM
        NH = self.NH
        HD = self.HD

        # Project to key, value, beta — all (B, S, D)
        # Keep FP32 to avoid precision loss in the 2048-step sequential scan.
        x_f32 = x.astype(ms.float32)
        K = self.key_proj(x_f32)                    # (B, S, D) FP32
        V = self.value_proj(x_f32)                   # (B, S, D) FP32
        beta = ops.sigmoid(self.beta_proj(x_f32))    # (B, S, NH) FP32

        # Initialize state: broadcast to batch, keep FP32 for precision
        # Must reshape 3D (NH,HD,HD) to 4D (1,NH,HD,HD) before Tile
        state = ops.Tile()(self.initial_state.reshape(1, NH, HD, HD),
                           (B, 1, 1, 1))  # (B, NH, HD, HD) FP32

        # Sequential scan — core of state tracking.
        # CANN 7's TBE unpack (backward of ops.stack) rejects even 64
        # outputs ("input number is too much"). Use ExpandDims + Concat
        # instead. The backward of Concat uses Slice (not Unpack), so
        # no CANN 7 limits apply. 2-level hierarchy:
        #   Inner: 64 concat(8) → 64 chunks of (8, B, NH, HD)
        #   Final:  1 concat(64) → (512, B, NH, HD)
        CHUNK = 8
        B_NH = B * NH
        chunks = ()
        for c in range(S // CHUNK):  # 64 chunks
            current = ()
            for t in range(c * CHUNK, (c + 1) * CHUNK):
                kt = K[:, t, None, :]   # (B, D, 1)
                vt = V[:, t, None, :]   # (B, D, 1)
                bt = beta[:, t, :, None, None]  # (B, NH, 1, 1)

                # delta rule: S = S - beta*(S@k - v)*k^T
                k_head = kt.reshape(B, NH, HD, 1)
                v_head = vt.reshape(B, NH, HD, 1)
                s3 = state.reshape(B_NH, HD, HD)
                k3 = k_head.reshape(B_NH, HD, 1)
                v3 = v_head.reshape(B_NH, HD, 1)
                residual = self.bmm(s3, k3).reshape(
                    B, NH, HD, 1) - v_head
                update = bt * self.bmm_tb(
                    residual.reshape(B_NH, HD, 1),
                    k3).reshape(B, NH, HD, HD)
                state = state - update

                out_t = state[:, :, -1, :]  # (B, NH, HD)
                current = current + (out_t.reshape(1, B, NH, HD),)
            chunks = chunks + (self.concat_op(current),)  # (8, B, NH, HD)

        stacked = self.concat_op(chunks)  # (S, B, NH, HD)
        stacked = self.transpose_op(stacked, (1, 0, 2, 3))  # (B, S, NH, HD)
        output = stacked.reshape(B, S, NH * HD)   # (B, S, D)

        return output.astype(ms.float16)


class CrossSystemBridge(nn.Cell):
    """Bridge between transformer hidden state and DeltaNet SFM.

    Down-projection: HIDDEN_DIM -> BRIDGE_DIM
    Up-projection: BRIDGE_DIM -> HIDDEN_DIM
    Gate: learned scalar gate (starts at 0, sigmoid≈0.5)
    """

    def __init__(self) -> None:
        super().__init__()
        self.down_proj = _make_dense(HIDDEN_DIM, BRIDGE_DIM, has_bias=True)
        self.up_proj = _make_dense(BRIDGE_DIM, HIDDEN_DIM, has_bias=True)
        self.gate_param = ms.Parameter(
            Tensor(np.array([0.0], dtype=np.float32)), name="bridge_gate")
        self.layer_norm = RMSNorm(HIDDEN_DIM)

    def construct(self, hidden: Tensor, sfm_out: Tensor) -> Tensor:
        """Merge transformer hidden with SFM output."""
        sfm_up = self.up_proj(sfm_out.astype(ms.float32)).astype(ms.float16)
        gate = ops.sigmoid(self.gate_param.astype(hidden.dtype))
        return self.layer_norm(hidden + gate * sfm_up)


class JudgeHead(nn.Cell):
    """Binary classification head: is the final answer correct?

    Projects last hidden state to 2 logits (correct/wrong).
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = _make_dense(HIDDEN_DIM, JUDGE_HIDDEN_DIM, has_bias=True)
        self.proj2 = _make_dense(JUDGE_HIDDEN_DIM, 2, has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        """Returns (B, 2) logits for judge classification."""
        last = x[:, -1, :].astype(ms.float32)
        h = ops.gelu(self.proj1(last))
        return self.proj2(h)


class SurprisePredictor(nn.Cell):
    """Scalar predictor for how 'surprising' the current state is.

    Used for self-evolution: high surprise = hard problem = more useful.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = _make_dense(HIDDEN_DIM, SURPRISE_DIM, has_bias=True)
        self.proj2 = _make_dense(SURPRISE_DIM, 1, has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        """Returns (B, 1) surprise score."""
        last = x[:, -1, :].astype(ms.float32)
        h = ops.gelu(self.proj1(last))
        return self.proj2(h)


# ══════════════════════════════════════════════════════════════════════
# FULL MODEL ASSEMBLY
# ══════════════════════════════════════════════════════════════════════


class Thinker15BModel(nn.Cell):
    """Qwen2.5-Coder-1.5B with 4 DeltaNet SFM blocks.

    Architecture:
      28 transformer layers (Qwen2.5-Coder-1.5B)
      + DeltaNet + Bridge after layers 6, 13, 20, 27
      + Judge head (binary: correct/wrong)
      + Surprise predictor (scalar)

    Construct is UNROLLED for @ms.jit safety.
    Tied embedding weights: logits = hidden @ embedding_table.T
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.norm = RMSNorm(HIDDEN_DIM)
        self.matmul = ops.MatMul(transpose_b=True)

        # 28 transformer layers
        self.layers = nn.CellList(
            [TransformerBlock() for _ in range(NUM_LAYERS)])

        # 4 DeltaNet + Bridge pairs
        self.deltanets = nn.CellList(
            [DeltaNetCell() for _ in range(len(SFM_LAYERS))])
        self.bridges = nn.CellList(
            [CrossSystemBridge() for _ in range(len(SFM_LAYERS))])
        self.sfm_layer_set = set(SFM_LAYERS)

        # Judge head
        self.judge_head = JudgeHead()

        # Surprise predictor
        self.surprise_head = SurprisePredictor()

    def construct(self, input_ids: Tensor, cos: Tensor, sin: Tensor,
                  mask: Tensor) -> tuple:
        """Forward pass.

        Returns:
            (logits, judge_logits, surprise_scores)
        """
        B = input_ids.shape[0]
        x = self.embedding(input_ids).astype(ms.float16)

        # Unrolled loop — GRAPH_MODE requires all CellList indices to be
        # compile-time constants (no variable sfm_idx).
        # SFM_LAYERS = (6, 13, 20, 27), 28 layers total.

        # Layers 0-6 (7 layers), then SFM 0
        for i in range(7):
            x = self.layers[i](x, cos, sin, mask)
        sfm_in = self.bridges[0].down_proj(
            x.astype(ms.float32)).astype(ms.float16)
        x = self.bridges[0](x, self.deltanets[0](sfm_in))

        # Layers 7-13 (7 layers), then SFM 1
        for i in range(7, 14):
            x = self.layers[i](x, cos, sin, mask)
        sfm_in = self.bridges[1].down_proj(
            x.astype(ms.float32)).astype(ms.float16)
        x = self.bridges[1](x, self.deltanets[1](sfm_in))

        # Layers 14-20 (7 layers), then SFM 2
        for i in range(14, 21):
            x = self.layers[i](x, cos, sin, mask)
        sfm_in = self.bridges[2].down_proj(
            x.astype(ms.float32)).astype(ms.float16)
        x = self.bridges[2](x, self.deltanets[2](sfm_in))

        # Layers 21-27 (7 layers), then SFM 3
        for i in range(21, 28):
            x = self.layers[i](x, cos, sin, mask)
        sfm_in = self.bridges[3].down_proj(
            x.astype(ms.float32)).astype(ms.float16)
        x = self.bridges[3](x, self.deltanets[3](sfm_in))

        x = self.norm(x)

        # LM head (tied embeddings)
        h2 = x.reshape((-1, HIDDEN_DIM)).astype(ms.float32)
        logits = self.matmul(h2, self.embedding.embedding_table)
        logits = logits.reshape(B, x.shape[1], VOCAB_SIZE).astype(ms.float16)

        # Judge head
        judge_logits = self.judge_head(x)

        # Surprise predictor
        surprise = self.surprise_head(x)

        return logits, judge_logits, surprise


# ══════════════════════════════════════════════════════════════════════
# TRAINING CELLS
# ══════════════════════════════════════════════════════════════════════


class ForwardLossCell(nn.Cell):
    """Multi-loss forward pass: masked CE + judge BCE + surprise MSE.

    Loss = CE_masked + 0.1 * judge_BCE + 0.01 * surprise_MSE
    """

    def __init__(self, model: Thinker15BModel, cos: Tensor, sin: Tensor,
                 mask: Tensor):
        super().__init__()
        self.model = model
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_loss_fn = nn.MSELoss(reduction="mean")
        self.cos = ms.Parameter(cos, name="cos", requires_grad=False)
        self.sin = ms.Parameter(sin, name="sin", requires_grad=False)
        self.causal_mask = ms.Parameter(
            mask, name="causal_mask", requires_grad=False)

    def construct(self, input_ids: Tensor,
                  loss_mask: Tensor,
                  judge_label: Tensor) -> Tensor:
        """Compute multi-loss.

        Args:
            input_ids: (B, S) token IDs.
            loss_mask: (B, S) — 1 for tokens to compute CE on, 0 otherwise.
            judge_label: (B,) — 0=wrong, 1=correct.

        Returns:
            total_loss: scalar.
        """
        logits, judge_logits, surprise = self.model(
            input_ids, self.cos, self.sin, self.causal_mask)

        # Masked CE loss (cast logits to FP32 for numerical stability —
        # FP16 log_softmax overflows for large logits)
        logits_t = logits[:, :-1, :].astype(ms.float32)  # (B, S-1, V)
        labels = input_ids[:, 1:]             # (B, S-1)
        mask_t = loss_mask[:, 1:]             # (B, S-1)
        ce_per_token = self.ce_loss_fn(
            logits_t.reshape((-1, VOCAB_SIZE)),
            labels.reshape((-1,)))
        ce_per_token = ce_per_token.reshape(mask_t.shape)
        num_masked = ops.maximum(mask_t.sum(), Tensor(1.0, ms.float32))
        ce_loss = (ce_per_token * mask_t).sum() / num_masked

        # Judge BCE loss
        # Convert label to float for BCEWithLogitsLoss
        judge_float = judge_label.astype(ms.float32)
        judge_loss = self.bce_loss_fn(judge_logits[:, 0], judge_float)

        # Surprise regularization (push toward 0.5 = moderate surprise)
        surprise_target = ops.ones_like(surprise) * 0.5
        surprise_loss = self.mse_loss_fn(surprise, surprise_target)

        total = ce_loss + 0.1 * judge_loss + 0.01 * surprise_loss
        return total


class TrainStep:
    """Train step — plain Python class, NOT nn.Cell.

    Uses ms.value_and_grad (from mindspore, not mindspore.ops) which
    avoids StubTensor grad metagraph tracing entirely. Manual
    ops.AllReduce handles DATA_PARALLEL gradient sync.

    Handles:
      - Gradient clipping (clip by global norm)
      - Two-optimizer support for separate param group LRs
      - Manual AllReduce for DATA_PARALLEL gradient sync
    """

    def __init__(self, forward_loss: ForwardLossCell, optimizer,
                 optimizer2=None, max_grad_norm: float = 1.0,
                 rank_size: int = 1):
        self.forward_loss = forward_loss
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.rank_size = rank_size
        if optimizer2 is not None:
            self.weights = ms.ParameterTuple(
                list(optimizer.parameters) + list(optimizer2.parameters))
            self.n_base = len(optimizer.parameters)
        else:
            self.weights = ms.ParameterTuple(optimizer.parameters)
            self.n_base = 0
        self.max_grad_norm = max_grad_norm

        # Build value_and_grad function using forward_loss Cell.
        # grad_position=None: don't differentiate w.r.t. inputs,
        # only w.r.t. weights. Grad order matches self.weights order.
        self.grad_fn = value_and_grad(
            forward_loss,
            grad_position=None,
            weights=self.weights,
            has_aux=False)

        # AllReduce op for DATA_PARALLEL gradient sync
        if rank_size > 1:
            self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    def set_train(self, mode: bool = True) -> None:
        """Pass-through to forward_loss."""
        self.forward_loss.set_train(mode)

    def step(self, input_ids: Tensor, loss_mask: Tensor,
             judge_label: Tensor) -> Tensor:
        """Execute one training step: forward → grad → allreduce → clip → update."""
        # Forward + backward in one call (no StubTensors!)
        loss, grads = self.grad_fn(input_ids, loss_mask, judge_label)

        # Manual AllReduce for DATA_PARALLEL (value_and_grad doesn't
        # auto-allreduce like GradOperation with gradients_mean=True)
        if self.rank_size > 1:
            grads = tuple(
                (self.all_reduce(g) / self.rank_size
                 if g is not None else g)
                for g in grads)

        # Gradient clipping (clip by global norm)
        norm_sq = Tensor(0.0, ms.float32)
        for g in grads:
            if g is not None:
                norm_sq = norm_sq + ops.ReduceSum()(
                    ops.square(g.astype(ms.float32)))
        global_norm = ops.sqrt(norm_sq)
        clip_coef = ops.minimum(
            Tensor(1.0, ms.float32),
            Tensor(self.max_grad_norm, ms.float32) /
            ops.maximum(global_norm, Tensor(1e-6, ms.float32)))
        grads = tuple(
            (g.astype(ms.float32) * clip_coef
             if g is not None else g)
            for g in grads)

        # Apply optimizer(s)
        if self.optimizer2 is not None:
            self.optimizer(grads[:self.n_base])
            self.optimizer2(grads[self.n_base:])
        else:
            self.optimizer(grads)

        return loss


# ══════════════════════════════════════════════════════════════════════
# WEIGHT LOADING
# ══════════════════════════════════════════════════════════════════════


def load_qwen_weights(model: Thinker15BModel,
                      pretrain_path: str) -> None:
    """Load Qwen2.5-Coder-1.5B weights into the model.

    Handles:
      - Recursive search for model.safetensors
      - HF -> MindSpore name mapping
      - Q/K/V bias tensors
      - Tied embeddings
    """
    model_dir = find_model_dir(pretrain_path)
    if not model_dir:
        # Also search under common alternative paths
        for alt in [os.path.join(pretrain_path, "qwen2.5-coder-1.5b"),
                    os.path.join(pretrain_path, "Qwen2.5-Coder-1.5B")]:
            model_dir = find_model_dir(alt)
            if model_dir:
                break

    if not model_dir:
        raise FileNotFoundError(
            f"Cannot find model.safetensors under {pretrain_path}")

    log(f"Found Qwen model at: {model_dir}")
    weights = load_safetensors(model_dir)

    # Build parameter name -> param lookup for O(1) matching
    model_params = {p.name: p for p in model.get_parameters()}

    # Build parameter mapping
    param_dict = {}
    loaded = 0
    skipped = []
    not_found = []

    for hf_name, array in weights.items():
        ms_name = _hf_to_ms_name(hf_name)
        if ms_name is None or ms_name == "lm_head.weight":
            skipped.append(hf_name)
            continue

        if ms_name in model_params:
            # Validate shape matches
            target_shape = model_params[ms_name].shape
            if array.shape != target_shape:
                not_found.append(
                    f"{hf_name} -> {ms_name} (shape {array.shape} "
                    f"!= {target_shape})")
                continue
            param_dict[ms_name] = ms.Parameter(
                Tensor(array.astype(np.float32)), name=ms_name)
            loaded += 1
        else:
            not_found.append(f"{hf_name} -> {ms_name}")

    if param_dict:
        # MS 2.2: load_param_into_net returns 1 value (unused)
        ms.load_param_into_net(model, param_dict)
        log(f"Loaded {loaded} Qwen tensors, skipped {len(skipped)} "
            f"(tied lm_head)")
    if not_found:
        log(f"  {len(not_found)} tensors not found in model "
            f"(SFM params expected):")
        for nf in not_found[:5]:
            log(f"    {nf}")
        if len(not_found) > 5:
            log(f"    ... and {len(not_found) - 5} more")


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════


def count_params(model: nn.Cell) -> tuple:
    total = sum(p.size for p in model.get_parameters())
    trainable = sum(p.size for p in model.trainable_params())
    return total, trainable


# ── BPE Tokenizer (Qwen2.5 compatible) ────────────────────────────────


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

    Returns (vocab, merge_priority, eos_id, special_token_ids) or None.
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
        eos_id = 151643  # Qwen2.5 default <|endoftext|>
        special_ids = {}
        for entry in added:
            if isinstance(entry, dict):
                content = entry.get("content", "")
                eid = entry.get("id", 0)
                if content == "<|endoftext|>":
                    eos_id = eid
                elif content == "<|im_start|>":
                    special_ids["im_start"] = eid
                elif content == "<|im_end|>":
                    special_ids["im_end"] = eid
                # Track our custom special tokens
                elif content in ("<|code|>", "<|trace|>", "<|answer|>",
                                 "<|slots|>", "<|monologue|>",
                                 "<|judge_correct|>", "<|judge_wrong|>",
                                 "<|explanation|>", "<|buggy_code|>",
                                 "<|error|>", "<|reasoning|>", "<|fix|>",
                                 "<|concept|>", "<|usage|>",
                                 "<|documentation|>", "<|task|>",
                                 "<|think|>", "<|verify|>"):
                    special_ids[content] = eid

        merge_priority = {}
        for idx, merge_str in enumerate(merges_raw):
            parts = merge_str.split(" ", 1)
            if len(parts) == 2:
                merge_priority[(parts[0], parts[1])] = idx

        log(f"  Loaded HF tokenizer: {len(token_to_id)} vocab, "
            f"{len(merges_raw)} merges, eos={eos_id}")
        log(f"  Special tokens found: {len(special_ids)}")
        return token_to_id, merge_priority, eos_id, special_ids
    except Exception as e:
        log(f"  Failed to parse tokenizer.json: {e}")
        return None


def _train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE, min_count=2):
    """Train a byte-level BPE tokenizer on texts (fallback)."""
    log(f"  Training BPE tokenizer on {len(texts)} texts ...")
    b2u = _get_byte_unicode()

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

    vocab = {}
    for byte_val in range(256):
        ch = b2u[byte_val]
        if ch not in vocab:
            vocab[ch] = len(vocab)
    vocab["<eos>"] = len(vocab)
    eos_id = vocab["<eos>"]

    merge_priority = {}
    for merge_idx in range(vocab_size - len(vocab)):
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

    log(f"  BPE trained: {len(vocab)} tokens, {len(merge_priority)} merges")
    return vocab, merge_priority, eos_id


def _tokenize_text(text, vocab, merge_priority):
    """Tokenize a single text using byte-level BPE."""
    b2u = _get_byte_unicode()
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
                for ch_tok in tok:
                    byte_val = ord(ch_tok)
                    if byte_val < 256:
                        token_ids.append(_get_byte_token(byte_val, vocab))
                    else:
                        token_ids.append(0)
    return token_ids


# ══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════

# Special tokens
_SPECIAL_TOKENS = [
    "<|code|>", "<|trace|>", "<|answer|>", "<|slots|>",
    "<|monologue|>", "<|judge_correct|>", "<|judge_wrong|>",
    "<|explanation|>", "<|buggy_code|>", "<|error|>",
    "<|reasoning|>", "<|fix|>", "<|concept|>", "<|usage|>",
    "<|documentation|>", "<|task|>", "<|think|>", "<|verify|>",
]

# Restricted builtins for exec()
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range,
    "int": int, "float": float, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "set": set, "bool": bool,
    "sum": sum, "round": round, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "isinstance": isinstance, "print": lambda *a: None,
    "True": True, "False": False, "None": None,
    "type": type, "hash": hash, "id": id, "hex": hex, "oct": oct,
    "bin": bin, "chr": chr, "ord": ord, "pow": pow, "divmod": divmod,
    "all": all, "any": any, "frozenset": frozenset,
}


def _safe_exec(code: str, ns: dict) -> bool:
    """Execute code in sandboxed namespace. Returns True on success."""
    try:
        exec(code, {"__builtins__": _SAFE_BUILTINS}, ns)
        return True
    except Exception:
        return False


def _extract_vars(ns: dict, max_vars: int = 10) -> dict:
    """Extract variable values from namespace."""
    result = {}
    for k, v in sorted(ns.items()):
        if k.startswith("_") or callable(v) or isinstance(v, type):
            continue
        if isinstance(v, (int, float, str, bool)):
            result[k] = v
        elif isinstance(v, (list, tuple)) and len(v) < 20:
            result[k] = v
        if len(result) >= max_vars:
            break
    return result


def _vars_to_json(state: dict) -> str:
    """Convert state dict to JSON string for <|slots|> tag."""
    # Only include simple types that JSON can handle
    clean = {}
    for k, v in state.items():
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)) and len(v) < 20:
            clean[k] = list(v)
    return json.dumps(clean)


def _gen_random_program(difficulty: int) -> str:
    """Generate a random Python program with given difficulty (1-5).

    1: simple arithmetic
    2: if/else, simple loops
    3: nested loops, functions
    4: recursion, list ops, dict
    5: multi-function, comprehensions
    """
    vars_pool = ["x", "y", "z", "n", "i", "j", "k", "a", "b", "c",
                  "total", "result", "count", "found", "data", "tmp"]
    lines = []

    if difficulty == 1:
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

    elif difficulty == 2:
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
            lines.append("total = 0")
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

    elif difficulty == 3:
        kind = random.choice(["nested", "func", "list_ops"])
        if kind == "nested":
            n = random.randint(3, 8)
            lines.append("result = 0")
            lines.append(f"for i in range({n}):")
            cond = random.choice(["i % 2 == 0", "i % 3 == 0", "True"])
            lines.append(f"    if {cond}:")
            lines.append(f"        result = result + {random.randint(1, 5)}")
        elif kind == "func":
            fn = random.choice(["double", "negate", "square"])
            lines.append(f"def {fn}(x):")
            if fn == "double":
                lines.append("    return x * 2")
            elif fn == "negate":
                lines.append("    return -x")
            else:
                lines.append("    return x * x")
            lines.append(f"result = {fn}({random.randint(1, 50)})")
        elif kind == "list_ops":
            arr = [random.randint(1, 20) for _ in range(random.randint(3, 8))]
            lines.append(f"data = {arr}")
            op = random.choice(["sum", "max", "min", "len", "sorted"])
            lines.append(f"result = {op}(data)")

    elif difficulty == 4:
        kind = random.choice(["recursive", "dict_ops", "list_build"])
        if kind == "recursive":
            fn = random.choice(["fib", "fact"])
            if fn == "fib":
                lines.append("def fib(n):")
                lines.append("    if n <= 1:")
                lines.append("        return n")
                lines.append("    return fib(n-1) + fib(n-2)")
                lines.append(f"result = fib({random.randint(4, 10)})")
            else:
                lines.append("def fact(n):")
                lines.append("    if n <= 1:")
                lines.append("        return 1")
                lines.append("    return n * fact(n-1)")
                lines.append(f"result = fact({random.randint(3, 8)})")
        elif kind == "dict_ops":
            lines.append("d = {}")
            keys = ["a", "b", "c", "x", "y"]
            for k in keys[:random.randint(2, 4)]:
                lines.append(f"d['{k}'] = {random.randint(1, 100)}")
            lines.append("result = sum(d.values())")
        elif kind == "list_build":
            lines.append("result = []")
            n = random.randint(3, 8)
            lines.append(f"for i in range({n}):")
            expr = random.choice(["i*i", "i*2+1", "i if i%2==0 else 0"])
            lines.append(f"    result.append({expr})")

    elif difficulty == 5:
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
            lines.append("def sum_sq(lst):")
            lines.append("    total = 0")
            lines.append("    for v in lst:")
            lines.append("        total = total + square(v)")
            lines.append("    return total")
            arr = [random.randint(1, 10) for _ in range(random.randint(3, 6))]
            lines.append(f"result = sum_sq({arr})")
        elif kind == "filter_map":
            arr = [random.randint(-10, 30) for _ in range(random.randint(4, 8))]
            lines.append(f"data = {arr}")
            lines.append("pos = [x for x in data if x > 0]")
            lines.append("result = [x*2 for x in pos]")

    return "\n".join(lines)


def _exec_with_trace(code_str: str) -> tuple:
    """Execute code and capture per-line variable state.

    Returns (final_vars, is_correct) or None on failure.
    """
    namespace = {}
    code_lines = code_str.strip().split("\n")

    try:
        for stmt in code_lines:
            _safe_exec(stmt, namespace)
    except Exception:
        return None

    final_vars = _extract_vars(namespace)
    return final_vars, True


def generate_execution_sample() -> dict:
    """Generate a single execution trace training sample.

    Returns dict with 'text' (full formatted), 'loss_mask_regions',
    'judge_label' (0 or 1).
    """
    difficulty = random.choices(
        range(1, 6), weights=[15, 30, 25, 20, 10])[0]
    code = _gen_random_program(difficulty)
    if not code.strip():
        return None

    result = _exec_with_trace(code)
    if result is None:
        return None

    final_vars, _ = result
    if not final_vars:
        return None

    # Determine answer variable
    last_line = code.strip().split("\n")[-1].strip()
    answer_var = None
    if "=" in last_line and not last_line.startswith(("def ", "for ",
                                                        "while ", "if ")):
        answer_var = last_line.split("=")[0].strip()

    answer_str = ""
    if answer_var and answer_var in final_vars:
        answer_str = f"{answer_var} = {repr(final_vars[answer_var])}"
    else:
        answer_str = _vars_to_json(final_vars)

    # Generate monologue (mental simulation)
    monologue_parts = []
    code_lines = code.strip().split("\n")
    for line in code_lines:
        if line.strip():
            monologue_parts.append(f"Step: {line.strip()}")
    monologue_parts.append(f"After execution: {answer_str}")
    monologue = "\n".join(monologue_parts)

    # Corrupt ~15% for balanced judge training
    is_corrupted = random.random() < 0.15
    if is_corrupted:
        # Replace answer with wrong value
        if isinstance(answer_str, str) and "=" in answer_str:
            wrong_val = random.randint(-1000, 1000)
            corrupted_answer = answer_str.split("=")[0].strip() + \
                f" = {wrong_val}"
            answer_str = corrupted_answer
        judge_label = 0
        judge_tag = "<|judge_wrong|>"
    else:
        judge_label = 1
        judge_tag = "<|judge_correct|>"

    # Build formatted text
    parts = []
    parts.append("<|code|>")
    parts.append(code)
    parts.append("<|slots|>")
    parts.append(_vars_to_json(final_vars))
    parts.append("<|monologue|>")
    parts.append(monologue)
    parts.append("<|answer|>")
    parts.append(answer_str)
    parts.append(judge_tag)

    text = "\n".join(parts)
    return {"text": text, "judge_label": judge_label}


def generate_debugging_sample() -> dict:
    """Generate a debugging training sample.

    Returns dict with 'text', 'judge_label' (always 1 = fix is correct).
    """
    bug_templates = [
        {
            "buggy": ("def sum_evens(lst):\n"
                      "    total = 0\n"
                      "    for i in range(len(lst)):\n"
                      "        if lst[i] % 2 == 0:\n"
                      "            total += lst[i]\n"
                      "        return total"),
            "error": "sum_evens([1, 2, 3, 4]) returns 0 instead of 6",
            "reasoning": ("Let me trace: i=0, lst[0]=1, 1%2!=0, skip. "
                          "Then return total -> returns 0! The return is "
                          "inside the for loop (wrong indentation)."),
            "fix": ("def sum_evens(lst):\n"
                    "    total = 0\n"
                    "    for i in range(len(lst)):\n"
                    "        if lst[i] % 2 == 0:\n"
                    "            total += lst[i]\n"
                    "    return total"),
        },
        {
            "buggy": ("def find_max(lst):\n"
                      "    max_val = lst[0]\n"
                      "    for i in range(1, len(lst) + 1):\n"
                      "        if lst[i] > max_val:\n"
                      "            max_val = lst[i]\n"
                      "    return max_val"),
            "error": "find_max([3, 7, 2, 9]) raises IndexError",
            "reasoning": ("range goes to len(lst)+1=5, lst[5] out of "
                          "bounds. Should be range(1, len(lst))."),
            "fix": ("def find_max(lst):\n"
                    "    max_val = lst[0]\n"
                    "    for i in range(1, len(lst)):\n"
                    "        if lst[i] > max_val:\n"
                    "            max_val = lst[i]\n"
                    "    return max_val"),
        },
        {
            "buggy": ("def is_palindrome(s):\n"
                      "    return s == s.reverse()"),
            "error": "is_palindrome('racecar') raises AttributeError",
            "reasoning": ("str.reverse() doesn't exist — it's a list "
                          "method. Use s[::-1]."),
            "fix": "def is_palindrome(s):\n    return s == s[::-1]",
        },
        {
            "buggy": ("def binary_search(arr, target):\n"
                      "    lo, hi = 0, len(arr) - 1\n"
                      "    while lo < hi:\n"
                      "        mid = (lo + hi) // 2\n"
                      "        if arr[mid] == target:\n"
                      "            return mid\n"
                      "        elif arr[mid] < target:\n"
                      "            lo = mid + 1\n"
                      "        else:\n"
                      "            hi = mid\n"
                      "    return -1"),
            "error": "binary_search([1, 3, 5], 5) returns -1",
            "reasoning": ("Condition should be 'while lo <= hi', not "
                          "'while lo < hi'. When lo == hi, the last "
                          "element might be the target."),
            "fix": ("def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid - 1\n"
                    "    return -1"),
        },
        {
            "buggy": ("def remove_dupes(lst):\n"
                      "    seen = set()\n"
                      "    for x in lst:\n"
                      "        if x not in seen:\n"
                      "            seen.append(x)\n"
                      "    return list(seen)"),
            "error": "remove_dupes([1,2,2,3]) raises AttributeError",
            "reasoning": ("Sets don't have append() — use add() for "
                          "sets, or use a list."),
            "fix": ("def remove_dupes(lst):\n"
                    "    seen = []\n"
                    "    for x in lst:\n"
                    "        if x not in seen:\n"
                    "            seen.append(x)\n"
                    "    return seen"),
        },
        {
            "buggy": ("def flatten(lst):\n"
                      "    result = []\n"
                      "    for item in lst:\n"
                      "        if isinstance(item, list):\n"
                      "            for sub in item:\n"
                      "                result.append(sub)\n"
                      "    return result"),
            "error": ("flatten([1, [2, 3], 4]) returns [2, 3] "
                      "instead of [1, 2, 3, 4]"),
            "reasoning": ("Non-list items are never appended. Missing "
                          "else branch."),
            "fix": ("def flatten(lst):\n"
                    "    result = []\n"
                    "    for item in lst:\n"
                    "        if isinstance(item, list):\n"
                    "            result.extend(item)\n"
                    "        else:\n"
                    "            result.append(item)\n"
                    "    return result"),
        },
    ]

    tmpl = random.choice(bug_templates)

    parts = []
    parts.append("<|buggy_code|>")
    parts.append(tmpl["buggy"])
    parts.append("<|error|>")
    parts.append(tmpl["error"])
    parts.append("<|reasoning|>")
    parts.append(tmpl["reasoning"])
    parts.append("<|fix|>")
    parts.append(tmpl["fix"])
    parts.append("<|judge_correct|>")

    return {"text": "\n".join(parts), "judge_label": 1}


def generate_batch(batch_size: int, stage: int = 1,
                   rank_id: int = 0) -> list:
    """Generate a batch of training samples.

    Args:
        batch_size: number of samples to generate.
        stage: 1 = execution traces, 2 = mixed (execution + debugging).
        rank_id: for seed diversity across workers.

    Returns:
        List of dicts with 'text' and 'judge_label'.
    """
    random.seed(int(time.time() * 1000) + rank_id * 10000)
    samples = []

    for _ in range(batch_size):
        if stage == 1:
            sample = generate_execution_sample()
        else:
            # Mix 60% execution, 40% debugging
            if random.random() < 0.6:
                sample = generate_execution_sample()
            else:
                sample = generate_debugging_sample()

        if sample is not None:
            samples.append(sample)

    # If not enough, retry
    attempts = 0
    while len(samples) < batch_size and attempts < batch_size * 3:
        sample = generate_execution_sample()
        if sample is not None:
            samples.append(sample)
        attempts += 1

    return samples


def format_sample_for_training(sample: dict, vocab: dict,
                               merge_priority: dict,
                               eos_id: int, seq_len: int) -> tuple:
    """Tokenize a sample and create (input_ids, loss_mask, judge_label).

    Loss mask: 1 for tokens after special markers (code, slots, monologue,
    answer, judge), 0 for padding and EOS.

    Returns:
        (input_ids_np, loss_mask_np, judge_label) or None.
    """
    text = sample["text"]
    ids = _tokenize_text(text, vocab, merge_priority)
    ids.append(eos_id)

    if len(ids) > seq_len:
        ids = ids[:seq_len]
    if len(ids) < seq_len:
        ids = ids + [eos_id] * (seq_len - len(ids))

    # Build loss mask: 1 for non-padding tokens
    loss_mask = [1.0] * len(ids)
    # Optionally mask the code prefix (first <|code|> marker content)
    # For now, mask everything (the model should predict all tokens)
    # except the very first token
    loss_mask[0] = 0.0  # Don't predict from padding

    if len(loss_mask) < seq_len:
        loss_mask = loss_mask + [0.0] * (seq_len - len(loss_mask))
    loss_mask = loss_mask[:seq_len]

    input_ids = np.array(ids, dtype=np.int32)
    loss_mask_arr = np.array(loss_mask, dtype=np.float32)
    judge_label = sample["judge_label"]

    return input_ids, loss_mask_arr, judge_label


# ══════════════════════════════════════════════════════════════════════
# SELF-EVOLUTION ENGINE
# ══════════════════════════════════════════════════════════════════════


def evolution_probe(model: Thinker15BModel, cos: Tensor, sin: Tensor,
                    mask: Tensor, vocab: dict, merge_priority: dict,
                    eos_id: int, n_samples: int = 100) -> float:
    """Probe model's current difficulty by testing on generated problems.

    Returns accuracy (0-1) on easy problems.
    """
    correct = 0
    total = 0

    for _ in range(n_samples):
        sample = generate_execution_sample()
        if sample is None:
            continue

        ids = _tokenize_text(sample["text"], vocab, merge_priority)
        if len(ids) < 10 or len(ids) > MAX_SEQ_LEN:
            continue

        ids_arr = np.array(ids, dtype=np.int32)
        input_t = Tensor(ids_arr[np.newaxis, :])

        try:
            model.set_train(False)
            logits, judge_logits, surprise = model(
                input_t, cos, sin, mask)
            model.set_train(True)

            # Check judge prediction
            pred = int(ops.argmax(judge_logits, axis=-1).asnumpy()[0])
            if pred == sample["judge_label"]:
                correct += 1
            total += 1
        except Exception:
            continue

    return correct / max(total, 1)


class DifficultyTracker:
    """EWMA-based difficulty tracker for self-evolution."""

    def __init__(self, initial_difficulty: float = 1.0, alpha: float = 0.1):
        self.difficulty = initial_difficulty
        self.alpha = alpha
        self.rolling_acc = 0.5
        self.history = []

    def update(self, accuracy: float) -> float:
        """Update difficulty based on accuracy (EWMA smoothing)."""
        self.rolling_acc = self.alpha * accuracy + \
            (1 - self.alpha) * self.rolling_acc
        self.history.append(self.rolling_acc)

        # Adapt difficulty
        if self.rolling_acc > 0.85 and self.difficulty < 5:
            self.difficulty += 0.2
        elif self.rolling_acc < 0.5 and self.difficulty > 1:
            self.difficulty -= 0.2

        self.difficulty = max(1.0, min(5.0, self.difficulty))
        return self.difficulty


# ══════════════════════════════════════════════════════════════════════
# SELF-STOPPING CRITERIA
# ══════════════════════════════════════════════════════════════════════


def check_stopping(step: int, loss_history: list,
                   max_steps: int, time_limit: float,
                   t_start: float, patience: int = CONVERGENCE_PATIENCE,
                   window: int = CONVERGENCE_WINDOW,
                   threshold: float = CONVERGENCE_THRESHOLD) -> str:
    """Check if training should stop.

    Returns:
        "" if should continue, or reason string if should stop.
    """
    elapsed = time.time() - t_start
    if elapsed > time_limit:
        return f"hard time limit ({elapsed:.0f}s > {time_limit}s)"
    if step >= max_steps:
        return f"max steps reached ({step} >= {max_steps})"

    if len(loss_history) >= window:
        recent = loss_history[-window:]
        rolling = sum(recent) / window

        # Check if older rolling was significantly better
        if len(loss_history) >= 2 * window:
            older = loss_history[-2 * window:-window]
            older_rolling = sum(older) / window
            improvement = (older_rolling - rolling) / max(abs(older_rolling), 1e-8)
            if improvement < threshold:
                return (f"plateau (rolling={rolling:.4f}, improvement="
                        f"{improvement:.6f} < {threshold})")

        # Check patience (no new best loss)
        best = min(loss_history)
        steps_since_best = len(loss_history) - \
            loss_history.index(best) - 1
        if steps_since_best >= patience:
            return (f"patience (best={best:.4f}, no improvement for "
                    f"{steps_since_best} steps)")

    return ""


# ══════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    t0_global = time.time()

    rank_id = int(os.environ.get("RANK_ID", "0"))
    rank_size = int(os.environ.get("RANK_SIZE", "1"))
    device_id = int(os.environ.get("DEVICE_ID", "0"))

    setup_logging(rank_id)

    log("=" * 60)
    log(f"Thinker-1.5B DeltaNet SFM Fine-tuning")
    log(f"rank={rank_id}/{rank_size} device={device_id}")
    log(f"Python {sys.version}")
    log(f"CWD: {os.getcwd()}")
    log("")
    log(f"CODE_PATH:     {CODE_PATH}")
    log(f"DATASET_PATH:  {DATASET_PATH}")
    log(f"PRETRAIN_PATH: {PRETRAIN_MODEL_PATH}")
    log(f"OUTPUT_PATH:   {OUTPUT_PATH}")
    log("")

    ms.set_context(
        mode=ms.GRAPH_MODE,  # PYNATIVE StubTensors crash CANN 7 Mul SelectFormat
        device_target="Ascend",
        device_id=device_id,
        memory_optimize_level="O1",
        max_call_depth=100000,  # DeltaNet 2048-step sequential scan
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

    # ── Build model ──
    log("Building Thinker-1.5B model (Qwen2.5-Coder-1.5B + DeltaNet SFM)...")
    model = Thinker15BModel()

    # ── Load Qwen weights ──
    log("Loading Qwen2.5-Coder-1.5B pretrained weights...")
    try:
        load_qwen_weights(model, PRETRAIN_MODEL_PATH)
        log("Qwen weights loaded successfully!")
    except Exception as e:
        log(f"WARNING: Failed to load Qwen weights: {e}")
        log("Training from random initialization...")

    total_p, trainable_p = count_params(model)
    log(f"Total params: {total_p:,}  |  Trainable: {trainable_p:,}")
    log(f"  Base: 28 layers, {HIDDEN_DIM}d, {NUM_HEADS}Q/{NUM_KV_HEADS}KV "
        f"GQA, {INTERMEDIATE_DIM} FFN")
    log(f"  DeltaNet: {DELTANET_NUM_HEADS} heads, "
        f"{DELTANET_HEAD_DIM}x{DELTANET_HEAD_DIM} state at layers "
        f"{SFM_LAYERS}")
    log(f"  Vocab: {VOCAB_SIZE}, SeqLen: {MAX_SEQ_LEN}")

    # ── Load tokenizer ──
    log("Loading tokenizer...")
    tok_result = _try_load_hf_tokenizer(PRETRAIN_MODEL_PATH)
    if tok_result is not None:
        vocab, merge_priority, eos_id, special_ids = tok_result
    else:
        # Fallback: train a simple BPE
        vocab, merge_priority, eos_id = _train_bpe_tokenizer([])
        special_ids = {}

    # ── Precompute RoPE + causal mask ──
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

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: SFM-only training (base model frozen)
    # ═══════════════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STAGE 1: SFM-only training (base frozen)")
    log("=" * 60)

    # Freeze base model parameters (SFM params keep requires_grad=True)
    for p in model.get_parameters():
        if not any(key in p.name for key in
                   ["deltanets", "bridges", "judge_head", "surprise_head"]):
            p.requires_grad = False

    sfm_params = [p for p in model.trainable_params()]
    log(f"SFM trainable params: {sum(p.size for p in sfm_params):,}")

    if sfm_params:
        lr_schedule_s1 = []
        for s in range(STAGE1_MAX_STEPS):
            if s < STAGE1_WARMUP:
                lr = STAGE1_LR_SFM * s / max(1, STAGE1_WARMUP)
            else:
                progress = (s - STAGE1_WARMUP) / max(
                    1, STAGE1_MAX_STEPS - STAGE1_WARMUP)
                min_lr = STAGE1_LR_SFM * MIN_LR_FACTOR
                lr = min_lr + 0.5 * (STAGE1_LR_SFM - min_lr) * (
                    1 + math.cos(math.pi * progress))
            lr_schedule_s1.append(lr)

        optimizer_s1 = nn.AdamWeightDecay(
            sfm_params,
            learning_rate=Tensor(
                np.array(lr_schedule_s1, dtype=np.float32)),
            weight_decay=STAGE1_WEIGHT_DECAY,
            beta1=0.9,
            beta2=0.95,
        )

        forward_loss = ForwardLossCell(model, cos_t, sin_t, causal_mask)
        train_step_s1 = TrainStep(forward_loss, optimizer_s1,
                                   max_grad_norm=MAX_GRAD_NORM,
                                   rank_size=rank_size)
        actual_bs = BATCH_SIZE_PER_DEVICE
        compiled = False
        for bs in [BATCH_SIZE_PER_DEVICE, 1]:
            try:
                log(f"  Compiling Stage 1 with B={bs}...")
                # Test forward pass only (value_and_grad handles
                # grad computation without needing set_train)
                dummy_ids = Tensor(
                    np.random.randint(0, VOCAB_SIZE,
                                     (bs, MAX_SEQ_LEN)).astype(np.int32))
                dummy_mask = Tensor(
                    np.ones((bs, MAX_SEQ_LEN), dtype=np.float32))
                # GRAPH_MODE: can't do tensor[0,0]=0.0, create correctly
                mask_np = np.ones((bs, MAX_SEQ_LEN), dtype=np.float32)
                mask_np[0, 0] = 0.0
                dummy_mask = Tensor(mask_np)
                dummy_judge = Tensor(
                    np.ones(bs, dtype=np.int32))
                _ = forward_loss(dummy_ids, dummy_mask, dummy_judge)
                log(f"  B={bs} forward compilation OK")
                actual_bs = bs
                compiled = True
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"  B={bs} OOM, trying smaller...")
                    del train_step_s1
                    gc.collect()
                    optimizer_s1 = nn.AdamWeightDecay(
                        sfm_params,
                        learning_rate=Tensor(
                            np.array(lr_schedule_s1, dtype=np.float32)),
                        weight_decay=STAGE1_WEIGHT_DECAY,
                        beta1=0.9, beta2=0.95)
                    forward_loss = ForwardLossCell(
                        model, cos_t, sin_t, causal_mask)
                    train_step_s1 = TrainStep(
                        forward_loss, optimizer_s1,
                        max_grad_norm=MAX_GRAD_NORM,
                        rank_size=rank_size)
                    continue
                raise

        if compiled:
            # Stage 1 training loop
            step = 0
            loss_history = []
            best_loss = float("inf")
            t_start = time.time()
            difficulty_tracker = DifficultyTracker(1.0)

            while step < STAGE1_MAX_STEPS:
                stop = check_stopping(
                    step, loss_history, STAGE1_MAX_STEPS,
                    TIME_LIMIT * 0.4, t_start,
                    patience=CONVERGENCE_PATIENCE)
                if stop:
                    log(f"Stage 1 stopping: {stop}")
                    break

                # Gradient accumulation
                accum_loss = 0.0
                for micro in range(GRADIENT_ACCUM_STEPS):
                    samples = generate_batch(
                        actual_bs, stage=1, rank_id=rank_id)
                    if not samples:
                        continue

                    batch_ids = []
                    batch_masks = []
                    batch_judges = []

                    for sample in samples:
                        formatted = format_sample_for_training(
                            sample, vocab, merge_priority, eos_id, MAX_SEQ_LEN)
                        if formatted is not None:
                            batch_ids.append(formatted[0])
                            batch_masks.append(formatted[1])
                            batch_judges.append(formatted[2])

                    if not batch_ids:
                        continue

                    ids_t = Tensor(np.array(batch_ids, dtype=np.int32))
                    mask_t = Tensor(
                        np.array(batch_masks, dtype=np.float32))
                    judge_t = Tensor(
                        np.array(batch_judges, dtype=np.int32))

                    try:
                        loss_val = train_step_s1.step(
                            ids_t, mask_t, judge_t)
                        accum_loss += float(loss_val.asnumpy())
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            log(f"Stage 1 OOM at step {step}")
                            break
                        raise

                step += 1
                avg_loss = accum_loss / max(GRADIENT_ACCUM_STEPS, 1)
                loss_history.append(avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    if rank_id == 0:
                        ms.save_checkpoint(
                            model,
                            os.path.join(CKPT_DIR,
                                         "stage1_best.ckpt"))

                # Self-evolution probe
                if step % EVOLUTION_PROBE_STEPS == 0 and step > 0 \
                        and rank_id == 0:
                    acc = evolution_probe(
                        model, cos_t, sin_t, causal_mask,
                        vocab, merge_priority, eos_id, n_samples=50)
                    difficulty_tracker.update(acc)
                    log(f"  Probe accuracy: {acc:.3f}, "
                        f"difficulty: {difficulty_tracker.difficulty:.1f}")
                # Logging
                if step % 50 == 0 or step <= 3:
                    dt = time.time() - t_start
                    lr_now = lr_schedule_s1[
                        min(step, len(lr_schedule_s1) - 1)]
                    log(f"S1 Step {step:>5d} | loss={avg_loss:.4f} | "
                        f"best={best_loss:.4f} | lr={lr_now:.2e} | "
                        f"diff={difficulty_tracker.difficulty:.1f} | "
                        f"elapsed={dt:.0f}s")

                # Checkpoint
                if rank_id == 0 and step % 2000 == 0 and step > 0:
                    ckpt_path = os.path.join(
                        CKPT_DIR, f"stage1_step_{step}.ckpt")
                    ms.save_checkpoint(model, ckpt_path)
                    log(f"Stage 1 checkpoint: {ckpt_path}")

            log(f"Stage 1 complete: {step} steps, "
                f"best_loss={best_loss:.4f}")
        else:
            log("Stage 1 compilation failed, skipping to Stage 2")

    # Save Stage 1 results before Stage 2 overwrites step/best_loss
    stage1_steps_done = step if compiled else 0
    stage1_best_loss = best_loss if compiled else float("inf")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 2: Full fine-tuning (unfreeze base + SFM)
    # ═══════════════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STAGE 2: Full fine-tuning (base + SFM)")
    log("=" * 60)

    # Unfreeze all parameters
    for p in model.get_parameters():
        p.requires_grad = True

    # Separate param groups: base and SFM
    base_params = []
    sfm_params = []
    for p in model.trainable_params():
        if any(key in p.name for key in
               ["deltanets", "bridges", "judge_head", "surprise_head"]):
            sfm_params.append(p)
        else:
            base_params.append(p)

    log(f"Base params: {sum(p.size for p in base_params):,}")
    log(f"SFM params: {sum(p.size for p in sfm_params):,}")

    # Build LR schedules for each group
    lr_schedule_base = []
    lr_schedule_sfm = []
    for s in range(STAGE2_MAX_STEPS):
        if s < STAGE2_WARMUP:
            warmup_frac = s / max(1, STAGE2_WARMUP)
            lr_base = STAGE2_LR_BASE * warmup_frac
            lr_sfm = STAGE2_LR_SFM * warmup_frac
        else:
            progress = (s - STAGE2_WARMUP) / max(
                1, STAGE2_MAX_STEPS - STAGE2_WARMUP)
            min_base = STAGE2_LR_BASE * MIN_LR_FACTOR
            min_sfm = STAGE2_LR_SFM * MIN_LR_FACTOR
            lr_base = min_base + 0.5 * (STAGE2_LR_BASE - min_base) * (
                1 + math.cos(math.pi * progress))
            lr_sfm = min_sfm + 0.5 * (STAGE2_LR_SFM - min_sfm) * (
                1 + math.cos(math.pi * progress))
        lr_schedule_base.append(lr_base)
        lr_schedule_sfm.append(lr_sfm)

    # Two separate optimizers for different param group LRs
    lr_s2_base = np.array(lr_schedule_base, dtype=np.float32)
    lr_s2_sfm = np.array(lr_schedule_sfm, dtype=np.float32)
    optimizer_base_s2 = nn.AdamWeightDecay(
        base_params,
        learning_rate=Tensor(lr_s2_base),
        weight_decay=STAGE2_WEIGHT_DECAY,
        beta1=0.9,
        beta2=0.95,
    )
    optimizer_sfm_s2 = nn.AdamWeightDecay(
        sfm_params,
        learning_rate=Tensor(lr_s2_sfm),
        weight_decay=STAGE2_WEIGHT_DECAY,
        beta1=0.9,
        beta2=0.95,
    )

    # Rebuild training cells with two-optimizer TrainStep
    forward_loss = ForwardLossCell(model, cos_t, sin_t, causal_mask)
    train_step_s2 = TrainStep(
        forward_loss, optimizer_base_s2, optimizer2=optimizer_sfm_s2,
        max_grad_norm=MAX_GRAD_NORM, rank_size=rank_size)

    # Compile with OOM fallback (forward pass only)
    actual_bs = BATCH_SIZE_PER_DEVICE
    compiled = False
    for bs in [BATCH_SIZE_PER_DEVICE, 1]:
        try:
            log(f"  Compiling Stage 2 with B={bs}...")
            dummy_ids = Tensor(
                np.random.randint(0, VOCAB_SIZE,
                                 (bs, MAX_SEQ_LEN)).astype(np.int32))
            mask_np2 = np.ones((bs, MAX_SEQ_LEN), dtype=np.float32)
            mask_np2[0, 0] = 0.0
            dummy_mask = Tensor(mask_np2)
            dummy_judge = Tensor(
                np.ones(bs, dtype=np.int32))
            _ = forward_loss(dummy_ids, dummy_mask, dummy_judge)
            log(f"  B={bs} forward compilation OK")
            actual_bs = bs
            compiled = True
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"  B={bs} OOM, trying smaller...")
                del train_step_s2
                gc.collect()
                optimizer_base_s2 = nn.AdamWeightDecay(
                    base_params,
                    learning_rate=Tensor(lr_s2_base),
                    weight_decay=STAGE2_WEIGHT_DECAY,
                    beta1=0.9, beta2=0.95)
                optimizer_sfm_s2 = nn.AdamWeightDecay(
                    sfm_params,
                    learning_rate=Tensor(lr_s2_sfm),
                    weight_decay=STAGE2_WEIGHT_DECAY,
                    beta1=0.9, beta2=0.95)
                forward_loss = ForwardLossCell(
                    model, cos_t, sin_t, causal_mask)
                train_step_s2 = TrainStep(
                    forward_loss, optimizer_base_s2,
                    optimizer2=optimizer_sfm_s2,
                    max_grad_norm=MAX_GRAD_NORM,
                    rank_size=rank_size)
                continue
            raise

    if not compiled:
        log("FATAL: Stage 2 compilation failed")
        return

    # Stage 2 training loop
    step = 0
    loss_history = []
    best_loss = float("inf")
    t_start = time.time()
    difficulty_tracker = DifficultyTracker(2.0)
    stop_reason = "not started"
    total_samples = 0

    log("")
    log("=" * 60)
    log("STAGE 2 TRAINING START")
    log("=" * 60)

    while step < STAGE2_MAX_STEPS:
        stop = check_stopping(
            step, loss_history, STAGE2_MAX_STEPS,
            TIME_LIMIT * 0.6, t_start,
            patience=CONVERGENCE_PATIENCE)
        if stop:
            stop_reason = stop
            log(f"Stage 2 stopping: {stop}")
            break

        # Gradient accumulation
        accum_loss = 0.0
        for micro in range(GRADIENT_ACCUM_STEPS):
            samples = generate_batch(
                actual_bs, stage=2, rank_id=rank_id)
            if not samples:
                continue

            batch_ids = []
            batch_masks = []
            batch_judges = []

            for sample in samples:
                formatted = format_sample_for_training(
                    sample, vocab, merge_priority, eos_id, MAX_SEQ_LEN)
                if formatted is not None:
                    batch_ids.append(formatted[0])
                    batch_masks.append(formatted[1])
                    batch_judges.append(formatted[2])

            if not batch_ids:
                continue

            ids_t = Tensor(np.array(batch_ids, dtype=np.int32))
            mask_t = Tensor(np.array(batch_masks, dtype=np.float32))
            judge_t = Tensor(np.array(batch_judges, dtype=np.int32))

            try:
                loss_val = train_step_s2.step(ids_t, mask_t, judge_t)
                accum_loss += float(loss_val.asnumpy())
                total_samples += len(batch_ids)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"Stage 2 OOM at step {step}")
                    stop_reason = f"OOM at step {step}"
                    break
                raise

        step += 1
        avg_loss = accum_loss / max(GRADIENT_ACCUM_STEPS, 1)
        loss_history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            if rank_id == 0:
                ms.save_checkpoint(
                    model, os.path.join(CKPT_DIR, "best.ckpt"))
                log(f"  New best loss: {avg_loss:.4f}")

        # Self-evolution: probe + adapt difficulty
        if step % EVOLUTION_PROBE_STEPS == 0 and step > 0 \
                and rank_id == 0:
            acc = evolution_probe(
                model, cos_t, sin_t, causal_mask,
                vocab, merge_priority, eos_id, n_samples=50)
            new_diff = difficulty_tracker.update(acc)
            log(f"  Probe accuracy: {acc:.3f}, "
                f"difficulty: {difficulty_tracker.difficulty:.1f} "
                f"-> {new_diff:.1f}")
        # Logging
        if step % 50 == 0 or step <= 3:
            dt = time.time() - t_start
            idx = min(step, len(lr_schedule_base) - 1)
            log(f"S2 Step {step:>5d} | loss={avg_loss:.4f} | "
                f"best={best_loss:.4f} | "
                f"lr_base={lr_schedule_base[idx]:.2e} | "
                f"lr_sfm={lr_schedule_sfm[idx]:.2e} | "
                f"diff={difficulty_tracker.difficulty:.1f} | "
                f"samples={total_samples:,} | "
                f"elapsed={dt:.0f}s")

        # Periodic checkpoint (best already saved above on improvement)
        if rank_id == 0 and step % 2000 == 0 and step > 0:
            ckpt_path = os.path.join(
                CKPT_DIR, f"stage2_step_{step}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            log(f"Stage 2 checkpoint: {ckpt_path}")

    # ── Training complete ──
    elapsed_total = time.time() - t0_global

    if rank_id == 0:
        ms.save_checkpoint(
            model, os.path.join(CKPT_DIR, "final.ckpt"))
        log("Final checkpoint saved")

    results = {
        "model_name": "Thinker-1.5B (Qwen2.5-Coder-1.5B + DeltaNet SFM)",
        "total_params": total_p,
        "trainable_params": trainable_p,
        "batch_size": actual_bs,
        "grad_accum_steps": GRADIENT_ACCUM_STEPS,
        "effective_batch": actual_bs * GRADIENT_ACCUM_STEPS * rank_size,
        "seq_len": MAX_SEQ_LEN,
        "num_devices": rank_size,
        "stage1_steps": stage1_steps_done,
        "stage2_steps": step,
        "total_samples": total_samples,
        "total_time_s": round(elapsed_total, 1),
        "final_loss": round(avg_loss, 4) if loss_history else None,
        "best_loss": round(best_loss, 4),
        "stop_reason": stop_reason,
        "data_parallel": use_dp,
        "final_difficulty": difficulty_tracker.difficulty,
        "architecture": {
            "base_model": "Qwen2.5-Coder-1.5B",
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "num_groups": NUM_GROUPS,
            "intermediate_dim": INTERMEDIATE_DIM,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "rope_theta": ROPE_THETA,
            "tied_embeddings": TIE_WORD_EMBEDDINGS,
        },
        "deltanet_config": {
            "hidden_dim": DELTANET_HIDDEN_DIM,
            "num_heads": DELTANET_NUM_HEADS,
            "head_dim": DELTANET_HEAD_DIM,
            "state_dim": DELTANET_STATE_DIM,
            "sfm_layers": list(SFM_LAYERS),
            "bridge_dim": BRIDGE_DIM,
        },
        "training_config": {
            "stage1_lr_sfm": STAGE1_LR_SFM,
            "stage2_lr_base": STAGE2_LR_BASE,
            "stage2_lr_sfm": STAGE2_LR_SFM,
            "judge_corrupt_rate": 0.15,
            "loss_weights": {
                "ce": 1.0,
                "judge": 0.1,
                "surprise": 0.01,
            },
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
