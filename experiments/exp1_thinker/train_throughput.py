"""
train_throughput.py - OpenI Training Throughput Benchmark for SFM + LoRA

Self-contained boot file for OpenI Train Task. Measures real training throughput
(tokens/sec) of Qwen2.5-Coder-7B-Instruct with LoRA + SFM adapters across
4x Ascend 910 ProA NPUs in DDP.

No imports from sfm/ — all architecture code is inline for OpenI portability.

Usage on OpenI:
    Set as boot file: experiments/exp1_thinker/train_throughput.py
    No command-line arguments needed — everything from c2net context.

    Local smoke test:
    python experiments/exp1_thinker/train_throughput.py --local_test
"""

# =============================================================================
# 0. Environment setup (BEFORE any torch imports)
# =============================================================================
import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"

# Line-buffering — wrap in try/except since reconfigure may fail on redirected
# stdout in some OpenI container setups.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Ascend performance env vars — MUST be set before torch/torch_npu import
os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
os.environ.setdefault("CPU_AFFINITY_CONF", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")

# HCCL init — must be set before torch_npu import
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

# =============================================================================
# 1. Logging
# =============================================================================
import logging

_logger_configured = False


def log(msg: str) -> None:
    """Print to stdout AND write to training.log (if configured)."""
    print(msg, flush=True)
    if _logger_configured:
        logging.info(msg)


def configure_file_logging(output_path: str) -> None:
    """Set up logging module to write to output_path/training.log."""
    global _logger_configured
    log_path = os.path.join(output_path, "training.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    _logger_configured = True
    log(f"File logging configured: {log_path}")


# =============================================================================
# 2. c2net platform integration (module-level, as required by c2net docs)
# =============================================================================
try:
    from c2net.context import prepare, upload_output
    _c2net_available = True
except ImportError:
    _c2net_available = False

    def prepare():
        class _DummyCtx:
            def __init__(self):
                self.output_path = "/tmp/sfm_throughput_output"
                self.pretrain_model_path = None
                self.model_path = None
        return _DummyCtx()

    def upload_output():
        pass

# =============================================================================
# 3. Rank info (from OpenI environment, before torch_npu)
# =============================================================================
_rank_id = int(os.environ.get("RANK_ID", os.environ.get("RANK", "0")))
_world_size = int(os.environ.get("RANK_SIZE", os.environ.get("WORLD_SIZE", "1")))
_device_id = int(os.environ.get("ASCEND_DEVICE_ID", os.environ.get("LOCAL_RANK", "0")))

# =============================================================================
# 4. c2net prepare() — module level (rank 0 first, then others)
# =============================================================================
_output_path = "/tmp/sfm_throughput_output"
_c2net_ctx = None
import time

if _rank_id == 0:
    log("Rank 0: calling c2net prepare()...")
    try:
        _c2net_ctx = prepare()
        _output_path = getattr(_c2net_ctx, "output_path", _output_path)
        os.makedirs(_output_path, exist_ok=True)
        configure_file_logging(_output_path)
    except Exception as e:
        log(f"Rank 0: c2net prepare() failed: {e}")

    # Signal other ranks
    _flag = "/cache/prepare_completed.txt"
    try:
        with open(_flag, "w") as _f:
            _f.write("0\n")
        log(f"Rank 0: prepare done, flag written")
    except OSError:
        pass
else:
    # Spin-wait for rank 0
    _flag = "/cache/prepare_completed.txt"
    for _ in range(60):
        if os.path.exists(_flag):
            break
        time.sleep(5)

    log(f"Rank {_rank_id}: calling c2net prepare()...")
    try:
        _c2net_ctx = prepare()
        _output_path = getattr(_c2net_ctx, "output_path", _output_path)
        os.makedirs(_output_path, exist_ok=True)
        configure_file_logging(_output_path)
    except Exception as e:
        log(f"Rank {_rank_id}: c2net prepare() failed: {e}")

# =============================================================================
# 5. PyTorch + torch_npu imports (after all env vars are set)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

try:
    import torch_npu
    _has_npu = True
except ImportError:
    _has_npu = False
    log("WARNING: torch_npu not available, running in CPU mode")

# =============================================================================
# 6. Distributed init
# =============================================================================
if _has_npu:
    _device = torch.device(f"npu:{_device_id}")
    torch.npu.set_device(_device)
else:
    _device = torch.device("cpu")

if _world_size > 1:
    try:
        dist.init_process_group(backend="hccl", world_size=_world_size, rank=_rank_id)
    except Exception:
        log("HCCL init failed, trying gloo fallback...")
        try:
            dist.init_process_group(backend="gloo", world_size=_world_size, rank=_rank_id)
        except Exception as e2:
            log(f"Distributed init failed completely: {e2}")
            _world_size = 1  # fall back to single-process

log(f"Rank {_rank_id}/{_world_size}, device={_device}")

if _rank_id == 0:
    log(f"torch:     {torch.__version__}")
    if _has_npu:
        log(f"torch_npu: {torch_npu.__version__}")
        try:
            _free_init, _total_hbm = torch.npu.mem_get_info(0)
            log(f"HBM/device: {_total_hbm / 1e9:.1f} GB, "
                f"free: {_free_init / 1e9:.1f} GB")
        except Exception as e:
            log(f"Could not read HBM info: {e}")


import argparse


# =============================================================================
# 7. Argument parsing (tolerant of unknown args from platform)
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFM+LoRA Throughput Benchmark")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--local_test", action="store_true")
    # parse_known_args silently ignores platform-injected args
    args, _ = parser.parse_known_args()
    return args


# =============================================================================
# 8. SFM Architecture Classes (self-contained)
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    output = base_linear(x) + (x @ A.T @ B.T) * (alpha / rank)
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 64, alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + (x @ self.A.T @ self.B.T) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.base_linear.bias

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features


class SFMSlotBank(nn.Module):
    """State Slot Bank — 16 slots x 256d with gated recurrent update.

    Cross-attention: hidden states (Q) attend over slot vectors (K/V).
    Update: s_t = sig(alpha)*s_{t-1} + sig(beta)*tanh(v)
    """

    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256,
                 num_slots: int = 16, num_heads: int = 4):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = slot_dim

        self.slot_vectors = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        self.q_proj = nn.Linear(hidden_dim, num_heads * slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, num_heads * slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, num_heads * slot_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * slot_dim, hidden_dim, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.W_alpha = nn.Linear(hidden_dim, slot_dim, bias=True)
        self.W_beta = nn.Linear(hidden_dim, slot_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, slot_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        if self.slot_vectors.dim() == 2:
            slots = self.slot_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            slots = self.slot_vectors

        Q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        K = self.k_proj(slots).view(
            batch_size, self.num_slots, self.num_heads, self.head_dim
        ).transpose(1, 2)
        V = self.v_proj(slots).view(
            batch_size, self.num_slots, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Manual SDPA for NPU compat
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        attn_output = self.out_proj(attn_output)

        modified_hidden = self.layer_norm(hidden_states + attn_output)

        h_pooled = hidden_states.mean(dim=1)
        alpha = torch.sigmoid(self.W_alpha(h_pooled))
        beta = torch.sigmoid(self.W_beta(h_pooled))
        v = torch.tanh(self.W_v(h_pooled))
        updated_slots = alpha.unsqueeze(1) * slots + beta.unsqueeze(1) * v.unsqueeze(1)

        return modified_hidden, updated_slots


class SFMAdapter(nn.Module):
    """Thin wrapper around SFMSlotBank for per-layer injection."""

    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256):
        super().__init__()
        self.slot_bank = SFMSlotBank(
            hidden_dim=hidden_dim, slot_dim=slot_dim,
            num_slots=16, num_heads=4,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        return self.slot_bank(hidden_states)


# =============================================================================
# 9. Model Injection
# =============================================================================

def inject_lora(model: nn.Module, rank: int = 64, alpha: float = 16.0) -> int:
    """Replace q/k/v/o_proj with LoRALinear wrappers. Returns count."""
    count = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, name):
                original = getattr(attn, name)
                if not isinstance(original, LoRALinear):
                    setattr(attn, name, LoRALinear(original, rank=rank, alpha=alpha))
                    count += 1
    return count


def inject_sfm_adapters(model: nn.Module, hidden_dim: int,
                        layer_indices: list) -> dict:
    """Inject SFM adapters at specified layers. Returns {idx: adapter}."""
    adapters = {}
    for idx in layer_indices:
        adapter = SFMAdapter(hidden_dim=hidden_dim, slot_dim=256)
        model.add_module(f"sfm_adapter_layer{idx}", adapter)

        def make_hook(adapter_ref):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                modified, _ = adapter_ref(hidden)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook

        model.model.layers[idx].register_forward_hook(make_hook(adapter))
        adapters[idx] = adapter
    return adapters


def count_parameters(model: nn.Module) -> tuple:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# 10. Throughput Benchmark
# =============================================================================

BENCHMARK_CONFIGS = [
    {"name": "A", "batch_size": 1, "seq_len": 2048},
    {"name": "B", "batch_size": 2, "seq_len": 2048},
    {"name": "C", "batch_size": 4, "seq_len": 2048},
    {"name": "D", "batch_size": 2, "seq_len": 4096},
    {"name": "E", "batch_size": 4, "seq_len": 1024},
    {"name": "F", "batch_size": 1, "seq_len": 8192},
]

WARMUP_STEPS = 5
MEASURE_STEPS = 15


def _npu_sync() -> None:
    try:
        torch.npu.synchronize()
    except AttributeError:
        pass


def run_one_config(model, optimizer, device, config) -> dict:
    """Run throughput benchmark for one config. Returns metrics dict or None on OOM."""
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]

    torch.manual_seed(42)
    try:
        torch.npu.manual_seed(42)
    except AttributeError:
        pass

    vocab_size = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    tokens_per_step = batch_size * seq_len

    log(f"  Config {config['name']}: B={batch_size}, S={seq_len}, "
        f"tok/step={tokens_per_step:,}")

    try:
        # Warmup
        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            _npu_sync()
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            _npu_sync()

        if _world_size > 1:
            dist.barrier()

        # Measure
        elapsed_list = []
        for _ in range(MEASURE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            _npu_sync()
            t0 = time.time()
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            _npu_sync()
            elapsed_list.append(time.time() - t0)

        if _world_size > 1:
            dist.barrier()

        avg_ms = sum(elapsed_list) / len(elapsed_list) * 1000
        tps_dev = tokens_per_step / (avg_ms / 1000)

        peak_hbm = free_hbm = 0.0
        try:
            peak_hbm = torch.npu.max_memory_allocated(device) / 1e9
            free_hbm = torch.npu.mem_get_info(device.index if device.index is not None else 0)[0] / 1e9
            torch.npu.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass

        return {
            "name": config["name"],
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "avg_ms": avg_ms,
            "tps_dev": tps_dev,
            "tps_total": tps_dev * _world_size,
            "peak_hbm": peak_hbm,
            "free_hbm": free_hbm,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"  Config {config['name']}: OOM (B={batch_size}, S={seq_len})")
            if _world_size > 1:
                dist.barrier()
            try:
                torch.npu.empty_cache()
            except AttributeError:
                pass
            return None
        raise


def print_results(results, output_path):
    """Print aligned results table. Rank 0 only."""
    log("")
    log("=" * 100)
    log("THROUGHPUT RESULTS")
    log("=" * 100)
    log(f"{'Cfg':>3} | {'Batch':>5} | {'SeqLen':>6} | {'Tok/step':>9} | "
        f"{'Tok/s/dev':>10} | {'Tok/s(total)':>12} | {'ms/step':>8} | "
        f"{'Peak HBM':>9} | {'Free HBM':>9}")
    log("-" * 100)

    for i, r in enumerate(results):
        cfg = BENCHMARK_CONFIGS[i]
        if r is not None:
            log(f"{r['name']:>3} | {r['batch_size']:>5} | {r['seq_len']:>6} | "
                f"{r['tokens_per_step']:>9,} | {r['tps_dev']:>10.1f} | "
                f"{r['tps_total']:>12.1f} | {r['avg_ms']:>8.1f} | "
                f"{r['peak_hbm']:>8.2f}G | {r['free_hbm']:>8.2f}G")
        else:
            log(f"{cfg['name']:>3} | B={cfg['batch_size']:<4} | S={cfg['seq_len']:<5} | OOM")

    valid = [r for r in results if r is not None]
    if valid:
        best = max(valid, key=lambda x: x["tps_dev"])
        log("")
        log(f"Best: {best['name']} (B={best['batch_size']}, S={best['seq_len']})")
        log(f"  {best['tps_dev']:.1f} tok/s/device, "
            f"{best['tps_total']:.1f} tok/s total ({_world_size}x NPUs)")
        hours = 1.1e9 / (best["tps_total"] * 3600)
        log(f"  1.1B tokens estimate: {hours:.1f} hours")

    # Save file
    results_file = os.path.join(output_path, "throughput_results.txt")
    with open(results_file, "w") as f:
        f.write("SFM+LoRA Training Throughput Benchmark\n")
        f.write(f"World size: {_world_size} NPUs\n\n")
        for i, r in enumerate(results):
            cfg = BENCHMARK_CONFIGS[i]
            if r is not None:
                f.write(f"Config {r['name']}: B={r['batch_size']}, S={r['seq_len']}\n"
                        f"  tok/s/dev={r['tps_dev']:.1f}, "
                        f"tok/s(total)={r['tps_total']:.1f}, "
                        f"ms/step={r['avg_ms']:.1f}\n"
                        f"  peak_hbm={r['peak_hbm']:.2f}GB, "
                        f"free_hbm={r['free_hbm']:.2f}GB\n\n")
            else:
                f.write(f"Config {cfg['name']}: B={cfg['batch_size']}, "
                        f"S={cfg['seq_len']} — OOM\n\n")
        if valid:
            best = max(valid, key=lambda x: x["tps_dev"])
            hours = 1.1e9 / (best["tps_total"] * 3600)
            f.write(f"Best: {best['name']} — {best['tps_dev']:.1f} tok/s/dev\n"
                    f"1.1B tokens estimate: {hours:.1f} hours\n")

    log(f"Results saved to {results_file}")


# =============================================================================
# 11. Main logic
# =============================================================================

def resolve_model_path(args):
    """Determine model path from args, c2net context, or auto-detect."""
    # 1. Explicit --model_path arg
    if args.model_path:
        return args.model_path

    # 2. c2net pretrain_model_path — auto-detect subfolder
    if _c2net_ctx is not None:
        pmp = getattr(_c2net_ctx, "pretrain_model_path", None)
        if pmp and os.path.isdir(pmp):
            # List subdirectories and pick the one that looks like a Qwen model
            try:
                entries = [e for e in os.listdir(pmp)
                           if os.path.isdir(os.path.join(pmp, e))]
                log(f"Model root: {pmp} — found: {entries}")
                # Prefer exact names in priority order
                for candidate in ["Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B"]:
                    if candidate in entries:
                        return os.path.join(pmp, candidate)
                # Fallback: use first subdirectory (assuming single model uploaded)
                if len(entries) == 1:
                    return os.path.join(pmp, entries[0])
                # Last resort: use root itself
                if os.path.exists(os.path.join(pmp, "config.json")):
                    return pmp
                log(f"WARNING: could not auto-detect model in {pmp}, "
                    f"subdirs: {entries}")
            except OSError as e:
                log(f"WARNING: could not list model dir: {e}")

        # 3. c2net model_path (alternative attribute)
        mp = getattr(_c2net_ctx, "model_path", None)
        if mp:
            return mp

    return None


def main():
    import traceback

    args = parse_args()

    # ---- Local test mode ----
    if args.local_test:
        log("LOCAL TEST MODE")
        h = torch.randn(1, 10, 3584)

        sb = SFMSlotBank(hidden_dim=3584, slot_dim=256)
        out, slots = sb(h)
        log(f"  SFMSlotBank: {h.shape} -> {out.shape}, slots {slots.shape}")

        ad = SFMAdapter(hidden_dim=3584)
        out2, slots2 = ad(h)
        log(f"  SFMAdapter:  {h.shape} -> {out2.shape}, slots {slots2.shape}")

        base = nn.Linear(512, 512)
        lora = LoRALinear(base, rank=8, alpha=16)
        y = lora(torch.randn(2, 8, 512))
        log(f"  LoRALinear:  (2,8,512) -> {y.shape}")

        log("\nLOCAL TEST COMPLETE")
        return

    # ---- Platform mode ----
    try:
        _run_benchmark(args)
    except Exception:
        tb = traceback.format_exc()
        log(f"\n{'='*80}")
        log("FATAL ERROR")
        log(f"{'='*80}")
        log(tb)
        # Save crash log and upload
        try:
            os.makedirs(_output_path, exist_ok=True)
            with open(os.path.join(_output_path, "CRASH_LOG.txt"), "w") as f:
                f.write(tb)
            log(f"Crash log saved to {_output_path}/CRASH_LOG.txt")
            try:
                upload_output()
                log("Partial results uploaded.")
            except Exception:
                pass
        except Exception:
            pass
        raise


def _run_benchmark(args):
    # Resolve model path
    model_path = resolve_model_path(args)
    if not model_path:
        log("ERROR: No model path found.")
        if _rank_id == 0:
            try:
                upload_output()
            except Exception:
                pass
        sys.exit(1)

    log(f"Model path: {model_path}")

    # Load model
    from transformers import AutoModelForCausalLM

    model_dtype = torch.float16

    if _rank_id == 0:
        log(f"Loading model (dtype={model_dtype})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model = model.to(_device)

    if _world_size > 1:
        dist.barrier()
    if _rank_id == 0:
        log("Model loaded on all ranks.")

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # SFM adapter layer indices
    num_layers = len(model.model.layers)
    sfm_indices = [num_layers // 4, num_layers // 2,
                   3 * num_layers // 4, num_layers - 1]
    hidden_dim = model.config.hidden_size

    # Inject adapters
    inject_sfm_adapters(model, hidden_dim, sfm_indices)
    lora_count = inject_lora(model, rank=64, alpha=16)

    # Move new modules to device
    for m in model.modules():
        if isinstance(m, (SFMAdapter, LoRALinear)):
            m.to(device=_device, dtype=model_dtype)

    if _world_size > 1:
        dist.barrier()

    if _rank_id == 0:
        total, trainable = count_parameters(model)
        log(f"Model: {model.config.model_type}, layers={num_layers}, "
            f"hidden={hidden_dim}")
        log(f"LoRA: rank=64, alpha=16, {lora_count} projections injected")
        log(f"SFM adapters: 4 at layers {sfm_indices}")
        log(f"Params: {total:,} total, {trainable:,} trainable "
            f"({100 * trainable / total:.2f}%)")

    # DDP
    if _world_size > 1:
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[_device_id], output_device=_device_id,
        )
    else:
        ddp_model = model

    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    if _world_size > 1:
        dist.barrier()

    # Benchmark
    if _rank_id == 0:
        log("")
        log("=" * 100)
        log("TRAINING THROUGHPUT BENCHMARK")
        log("=" * 100)
        log(f"Configs: {len(BENCHMARK_CONFIGS)}, "
            f"Warmup: {WARMUP_STEPS}, Measure: {MEASURE_STEPS}")
        log("")

    results = []
    for cfg in BENCHMARK_CONFIGS:
        results.append(run_one_config(ddp_model, optimizer, _device, cfg))

    # Results (rank 0)
    if _rank_id == 0:
        print_results(results, _output_path)
        log("\nUploading results...")
        try:
            upload_output()
            log("Upload complete.")
        except Exception as e:
            log(f"Upload failed (non-fatal): {e}")

    # Cleanup
    if _world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if _rank_id == 0:
        log("\n" + "=" * 100)
        log("DONE")
        log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
