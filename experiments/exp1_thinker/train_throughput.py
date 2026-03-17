"""
train_throughput.py - OpenI Training Throughput Benchmark for SFM + LoRA

Self-contained boot file for OpenI Train Task. Measures real training throughput
(tokens/sec) of Qwen2.5-Coder-7B with LoRA + SFM adapters across NPUs in DDP.

Designed for the mindspore_2_2_cann7_train image: installs PyTorch + torch_npu
via pip at startup (container has internet + CANN 7 drivers, just needs packages).

No imports from sfm/ — all architecture code is inline for OpenI portability.

Usage:
    # OpenI Train Task: set as boot file, no args needed
    # Local test: python train_throughput.py --local_test
"""

# ===========================================================================
# Phase 0: Standard lib imports + diagnostics (NO torch yet)
# ===========================================================================
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# --- Early boot log: /tmp/sfm_boot.log (before c2net prepare()) ---
# This ensures output is captured even if pip install or early init fails.
_boot_log = "/tmp/sfm_boot.log"
_log_file = _boot_log  # global used by log()
try:
    with open(_boot_log, "w") as f:
        f.write(f"[BOOT] train_throughput.py starting\n")
        f.write(f"[BOOT] Python {sys.version}\n")
        f.write(f"[BOOT] CWD: {os.getcwd()}\n")
        f.write(f"[BOOT] RANK_ID={os.getenv('RANK_ID', 'NOT SET')}\n")
except Exception:
    pass

def log(msg: str) -> None:
    """Print to stdout AND write to log file (if set)."""
    print(msg, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

log("=" * 60)
log("[BOOT] train_throughput.py starting")
log(f"[BOOT] Python {sys.version}")
log(f"[BOOT] CWD: {os.getcwd()}")
log(f"[BOOT] RANK_ID={os.getenv('RANK_ID', 'NOT SET')}")


# ===========================================================================
# Phase 1: Install PyTorch + torch_npu + transformers if missing
# ===========================================================================

_RANK_ID = os.getenv("RANK_ID", "0")
_RANK_ID = int(_RANK_ID) if _RANK_ID.isdigit() else 0
_INSTALL_DONE = "/cache/pip_install_done.txt"


def _pip_install(*packages, timeout=300):
    """Install packages via pip. Uses system default index (no mirror override).
    Streams output and has a timeout to prevent hanging."""
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install"] + list(packages) + ["-q"]
    log(f"[PIP] pip install {' '.join(packages)} (timeout={timeout}s)")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, b''):
            text = line.decode('utf-8', errors='replace').strip()
            if text:
                log(f"  {text}")
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"[PIP] TIMEOUT after {timeout}s — killing pip")
        p.kill()
        p.wait()
        return False
    if p.returncode != 0:
        log(f"[PIP] FAILED (exit code {p.returncode})")
    return p.returncode == 0


def _try_import(name):
    """Try importing a module. Returns True on success."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# --- Install gate: only rank 0 installs, others wait ---
# OpenI launches 4 copies of this script (one per NPU). If all 4 run pip
# install simultaneously writing to the same site-packages, shared library
# files get corrupted ("ELF load command past end of file").
# Fix: rank 0 installs and writes a done file; others wait for it.

_is_local_test = "--local_test" in sys.argv

if _is_local_test or _RANK_ID == 0:
    _install_ok = True

    # Check if torch/torch_npu are already working (not corrupted)
    if _try_import("torch") and (_is_local_test or _try_import("torch_npu")):
        import torch
        log(f"[BOOT] torch {torch.__version__} (pre-installed, working)")
        if _try_import("torch_npu"):
            import torch_npu
            log(f"[BOOT] torch_npu {torch_npu.__version__} (pre-installed, working)")
    else:
        # Previous runs may have corrupted torch/torch_npu (parallel pip).
        # Uninstall first to ensure a clean state.
        log("[BOOT] Cleaning up any corrupted torch/torch_npu from previous runs...")
        _pip_install("uninstall", "torch", "torch_npu", "-y", timeout=60)

        # Try CANN 7 compatible versions first, then latest as fallback
        installed = False
        for attempt_name, torch_spec, npu_spec in [
            ("CANN 7 exact", "torch==2.1.0", "torch_npu==2.1.0"),
            ("latest", "torch", "torch_npu"),
        ]:
            log(f"[BOOT] Trying {attempt_name}: {torch_spec} + {npu_spec}")
            pkgs = [torch_spec]
            if not _is_local_test:
                pkgs.append(npu_spec)
            if not _pip_install(*pkgs, timeout=600):
                log(f"[BOOT] {attempt_name}: pip install failed")
                continue
            # Verify imports
            try:
                import torch
                log(f"[BOOT] torch {torch.__version__} imported OK")
            except ImportError as e:
                log(f"[BOOT] torch import FAILED: {e}")
                continue
            if _is_local_test:
                installed = True
                break
            try:
                import torch_npu
                log(f"[BOOT] torch_npu {torch_npu.__version__} imported OK")
                installed = True
                break
            except ImportError as e:
                log(f"[BOOT] torch_npu import FAILED: {e}")
                log(f"[BOOT]   (CANN version mismatch, trying next)")
        if not installed:
            log("[BOOT] FATAL: no working torch+torch_npu combination found.")
            _install_ok = False

    # Install transformers + c2net (rank 0 only)
    if _install_ok and not _try_import("transformers"):
        log("[BOOT] Installing transformers...")
        if not _pip_install("transformers"):
            log("[BOOT] FATAL: cannot install transformers")
            _install_ok = False
        else:
            log("[BOOT] transformers installed OK")

    if _install_ok and not _try_import("c2net"):
        log("[BOOT] Installing c2net...")
        _pip_install("-U", "c2net")
        if not _try_import("c2net"):
            log("[BOOT] FATAL: cannot install c2net")
            _install_ok = False
        else:
            log("[BOOT] c2net installed OK")

    # ALWAYS write done file (even on failure) so other ranks don't hang
    if not _is_local_test:
        try:
            with open(_INSTALL_DONE, "w") as f:
                f.write("done\n" if _install_ok else "FAILED\n")
            status = "SUCCESS" if _install_ok else "FAILED"
            log(f"[BOOT] Rank 0: install {status}, wrote {_INSTALL_DONE}")
        except Exception as e:
            log(f"[BOOT] Rank 0: failed to write done file: {e}")

    if not _install_ok:
        sys.exit(1)
else:
    # Non-zero ranks: wait for rank 0 to finish installing
    log(f"[BOOT] Rank {_RANK_ID}: waiting for rank 0 to finish pip install...")
    waited = 0
    while not os.path.exists(_INSTALL_DONE):
        time.sleep(5)
        waited += 5
        if waited % 30 == 0:
            log(f"[BOOT] Rank {_RANK_ID}: still waiting ({waited}s)...")
        if waited > 900:
            log(f"[BOOT] Rank {_RANK_ID}: pip install wait TIMEOUT (15 min)")
            sys.exit(1)
    # Check if rank 0 succeeded or failed
    try:
        with open(_INSTALL_DONE, "r") as f:
            status = f.read().strip()
    except Exception:
        status = "unknown"
    if status == "FAILED":
        log(f"[BOOT] Rank {_RANK_ID}: rank 0 install FAILED, exiting")
        sys.exit(1)
    log(f"[BOOT] Rank {_RANK_ID}: rank 0 install OK ({waited}s)")


# ===========================================================================
# Phase 2: Now safe to import torch ecosystem
# ===========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from c2net.context import prepare, upload_output

_has_npu = _try_import("torch_npu")
if _has_npu:
    import torch_npu
    log(f"[BOOT] All dependencies ready. "
        f"torch={torch.__version__}, torch_npu={torch_npu.__version__}")
else:
    log(f"[BOOT] torch={torch.__version__} ready "
        f"(torch_npu skipped for local test)")


# ===========================================================================
# Phase 3: Args
# ===========================================================================
parser = argparse.ArgumentParser(description="SFM+LoRA Throughput Benchmark")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--local_test", action="store_true")
parser.add_argument("--npu_count", type=int, default=0)


# ===========================================================================
# Phase 5: SFM Architecture Classes
# ===========================================================================

class LoRALinear(nn.Module):
    """output = base(x) + (x @ A.T @ B.T) * scale"""

    def __init__(self, base_linear: nn.Linear, rank: int = 64, alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        self.scaling = alpha / rank
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + (x @ self.A.T @ self.B.T) * self.scaling

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    @property
    def in_features(self):
        return self.base_linear.in_features

    @property
    def out_features(self):
        return self.base_linear.out_features


class SFMSlotBank(nn.Module):
    """16 slots x 256d with cross-attention and gated recurrent update."""

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
        self.W_alpha = nn.Linear(hidden_dim, slot_dim)
        self.W_beta = nn.Linear(hidden_dim, slot_dim)
        self.W_v = nn.Linear(hidden_dim, slot_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        B, S, _ = hidden_states.shape
        slots = self.slot_vectors.unsqueeze(0).expand(B, -1, -1)
        Q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(slots).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, -1)
        out = self.out_proj(out)
        modified = self.layer_norm(hidden_states + out)
        h = hidden_states.mean(dim=1)
        a = torch.sigmoid(self.W_alpha(h))
        b = torch.sigmoid(self.W_beta(h))
        v = torch.tanh(self.W_v(h))
        new_slots = a.unsqueeze(1) * slots + b.unsqueeze(1) * v.unsqueeze(1)
        return modified, new_slots


class SFMAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256):
        super().__init__()
        self.slot_bank = SFMSlotBank(hidden_dim, slot_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        return self.slot_bank(hidden_states)


# ===========================================================================
# Phase 6: Injection
# ===========================================================================

def inject_lora(model: nn.Module, rank: int = 64, alpha: float = 16.0) -> int:
    count = 0
    for layer in model.model.layers:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(layer.self_attn, name, None)
            if orig is not None and not isinstance(orig, LoRALinear):
                setattr(layer.self_attn, name, LoRALinear(orig, rank, alpha))
                count += 1
    return count


def inject_sfm_adapters(model: nn.Module, hidden_dim: int, indices: list) -> dict:
    adapters = {}
    for idx in indices:
        adapter = SFMAdapter(hidden_dim, slot_dim=256)
        model.add_module(f"sfm_adapter_layer{idx}", adapter)
        def _hook(ref):
            def hook(module, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                modified, _ = ref(hidden)
                return (modified,) + output[1:] if isinstance(output, tuple) else modified
            return hook
        model.model.layers[idx].register_forward_hook(_hook(adapter))
        adapters[idx] = adapter
    return adapters


def count_params(model: nn.Module) -> tuple:
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, tr


# ===========================================================================
# Phase 7: Model path resolution
# ===========================================================================

def resolve_model_path(c2net_ctx, args):
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


# ===========================================================================
# Phase 8: Benchmark
# ===========================================================================

CONFIGS = [
    {"name": "A", "batch_size": 1, "seq_len": 2048},
    {"name": "B", "batch_size": 2, "seq_len": 2048},
    {"name": "C", "batch_size": 4, "seq_len": 2048},
    {"name": "D", "batch_size": 2, "seq_len": 4096},
    {"name": "E", "batch_size": 4, "seq_len": 1024},
    {"name": "F", "batch_size": 1, "seq_len": 8192},
]
WARMUP = 5
MEASURE = 15


def _sync():
    if _has_npu:
        try:
            torch.npu.synchronize()
        except AttributeError:
            pass


def run_one_config(model, optimizer, device, cfg, rank, world_size, dist):
    B, S = cfg["batch_size"], cfg["seq_len"]
    tok = B * S
    torch.manual_seed(42)
    try:
        torch.npu.manual_seed(42)
    except AttributeError:
        pass
    vocab = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size
    input_ids = torch.randint(0, vocab, (B, S), device=device)
    labels = input_ids.clone()
    log(f"  Config {cfg['name']}: B={B}, S={S}, tok/step={tok:,}")
    try:
        for _ in range(WARMUP):
            optimizer.zero_grad(set_to_none=True)
            _sync()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            _sync()
        if world_size > 1:
            dist.barrier()
        elapsed = []
        for _ in range(MEASURE):
            optimizer.zero_grad(set_to_none=True)
            _sync()
            t0 = time.time()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            _sync()
            elapsed.append(time.time() - t0)
        if world_size > 1:
            dist.barrier()
        avg_ms = sum(elapsed) / len(elapsed) * 1000
        tps = tok / (avg_ms / 1000)
        peak = free = 0.0
        try:
            peak = torch.npu.max_memory_allocated(device) / 1e9
            free = torch.npu.mem_get_info(device.index if device.index is not None else 0)[0] / 1e9
            torch.npu.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass
        return {"name": cfg["name"], "batch_size": B, "seq_len": S,
                "tok": tok, "avg_ms": avg_ms,
                "tps_dev": tps, "tps_total": tps * world_size,
                "peak": peak, "free": free}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"  Config {cfg['name']}: OOM (B={B}, S={S})")
            if world_size > 1:
                dist.barrier()
            try:
                torch.npu.empty_cache()
            except AttributeError:
                pass
            return None
        raise


def print_results(results, world_size, output_path):
    log("")
    log("=" * 100)
    log("THROUGHPUT RESULTS")
    log("=" * 100)
    log(f"{'Cfg':>3} | {'Batch':>5} | {'SeqLen':>6} | {'Tok/step':>9} | "
        f"{'Tok/s/dev':>10} | {'Tok/s(total)':>12} | {'ms/step':>8} | "
        f"{'Peak HBM':>9} | {'Free HBM':>9}")
    log("-" * 100)
    for i, r in enumerate(results):
        cfg = CONFIGS[i]
        if r is not None:
            log(f"{r['name']:>3} | {r['batch_size']:>5} | {r['seq_len']:>6} | "
                f"{r['tok']:>9,} | {r['tps_dev']:>10.1f} | "
                f"{r['tps_total']:>12.1f} | {r['avg_ms']:>8.1f} | "
                f"{r['peak']:>8.2f}G | {r['free']:>8.2f}G")
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
    path = os.path.join(output_path, "throughput_results.txt")
    with open(path, "w") as f:
        f.write("SFM+LoRA Training Throughput Benchmark\n")
        f.write(f"World size: {world_size} NPUs\n\n")
        for i, r in enumerate(results):
            cfg = CONFIGS[i]
            if r is not None:
                f.write(f"Config {r['name']}: B={r['batch_size']}, S={r['seq_len']}\n"
                        f"  tok/s/dev={r['tps_dev']:.1f}, tok/s(total)={r['tps_total']:.1f}\n"
                        f"  ms/step={r['avg_ms']:.1f}, peak={r['peak']:.2f}GB, free={r['free']:.2f}GB\n\n")
            else:
                f.write(f"Config {cfg['name']}: OOM\n\n")
        if valid:
            best = max(valid, key=lambda x: x["tps_dev"])
            hours = 1.1e9 / (best["tps_total"] * 3600)
            f.write(f"Best: {best['name']} — {best['tps_dev']:.1f} tok/s/dev\n"
                    f"1.1B tokens: {hours:.1f} hours\n")
    log(f"Results saved to {path}")


# ===========================================================================
# Phase 9: Local test
# ===========================================================================

def run_local_test():
    log("LOCAL TEST MODE")
    h = torch.randn(1, 10, 3584)
    sb = SFMSlotBank(3584, 256)
    o, s = sb(h)
    log(f"  SFMSlotBank: {h.shape} -> {o.shape}, slots {s.shape}")
    ad = SFMAdapter(3584)
    o2, s2 = ad(h)
    log(f"  SFMAdapter:  {h.shape} -> {o2.shape}, slots {s2.shape}")
    base = nn.Linear(512, 512)
    lora = LoRALinear(base, rank=8, alpha=16)
    y = lora(torch.randn(2, 8, 512))
    log(f"  LoRALinear:  (2,8,512) -> {y.shape}")
    log("\nLOCAL TEST COMPLETE")


# ===========================================================================
# Phase 10: Main
# ===========================================================================

def main():
    global _log_file
    args, unknown = parser.parse_known_args()

    if args.local_test:
        run_local_test()
        return

    rank_id_str = os.getenv('RANK_ID')
    if rank_id_str is None:
        rank_id = 0
        is_multi_card = False
    else:
        rank_id = int(rank_id_str)
        is_multi_card = True

    log(f"Rank {rank_id}, multi_card={is_multi_card}")

    # ================================================================
    # Phase A: Data preparation (rank 0 only, others wait)
    # ================================================================
    _SYNC_FILE = "/cache/sfm_ready.txt"
    model_path = None
    output_path = None
    world_size = 1

    if rank_id == 0:
        log("Rank 0: calling c2net prepare()...")
        try:
            c2net_ctx = prepare()
            log(f"Rank 0: prepare() OK")
            log(f"  output_path:         {getattr(c2net_ctx, 'output_path', 'N/A')}")
            log(f"  pretrain_model_path: {getattr(c2net_ctx, 'pretrain_model_path', 'N/A')}")
            log(f"  dataset_path:        {getattr(c2net_ctx, 'dataset_path', 'N/A')}")

            output_path = c2net_ctx.output_path
            os.makedirs(output_path, exist_ok=True)
            _log_file = os.path.join(output_path, "training.log")
            # Copy boot log into output dir so it gets uploaded
            if os.path.exists(_boot_log):
                try:
                    import shutil
                    shutil.copy2(_boot_log, os.path.join(output_path, "boot.log"))
                except Exception:
                    pass

            model_path = resolve_model_path(c2net_ctx, args)
            if not model_path:
                log("ERROR: no model path found.")
                with open(_SYNC_FILE, "w") as f:
                    f.write("ERROR\n")
                sys.exit(1)

            # Detect NPU count
            try:
                detected = torch.npu.device_count()
                log(f"Rank 0: NPU count={detected}")
            except Exception as e:
                log(f"Rank 0: torch.npu.device_count() failed: {e}")
                detected = 1

            world_size = args.npu_count if args.npu_count > 0 else detected
            log(f"Rank 0: model={model_path}, world_size={world_size}")

            with open(_SYNC_FILE, "w") as f:
                f.write(f"{output_path}\n{model_path}\n{world_size}\n")
            log("Rank 0: sync file written.")

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

    if rank_id != 0:
        with open(_SYNC_FILE, "r") as f:
            output_path = f.readline().strip()
            model_path = f.readline().strip()
            world_size = int(f.readline().strip())
        os.makedirs(output_path, exist_ok=True)
        _log_file = os.path.join(output_path, "training.log")
        log(f"Rank {rank_id}: model={model_path}, world_size={world_size}")

    # ================================================================
    # Phase B: Device setup
    # ================================================================
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
    os.environ.setdefault("CPU_AFFINITY_CONF", "1")
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(rank_id)

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    log(f"Rank {rank_id}: npu:0 ready")

    if rank_id == 0:
        log(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}")
        try:
            free_i, total_i = torch.npu.mem_get_info(0)
            log(f"HBM: {total_i/1e9:.1f} GB total, {free_i/1e9:.1f} GB free")
        except Exception as e:
            log(f"HBM info: {e}")

    # ================================================================
    # Phase C: DDP
    # ================================================================
    import torch.distributed as dist

    if world_size > 1:
        try:
            dist.init_process_group(backend="hccl",
                                    rank=rank_id, world_size=world_size)
            log(f"Rank {rank_id}: DDP OK (hccl, {world_size} NPUs)")
        except Exception as e:
            log(f"Rank {rank_id}: DDP failed: {e}, falling back to single-card")
            world_size = 1

    # ================================================================
    # Phase D: Load model
    # ================================================================
    if rank_id == 0:
        log(f"Loading model: {model_path}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.to(device)

    if world_size > 1:
        dist.barrier()
    if rank_id == 0:
        log("Model loaded on all ranks.")

    # ================================================================
    # Phase E: Freeze + inject
    # ================================================================
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    sfm_idx = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    h_dim = model.config.hidden_size

    inject_sfm_adapters(model, h_dim, sfm_idx)
    lora_n = inject_lora(model, rank=64, alpha=16)

    for m in model.modules():
        if isinstance(m, (SFMAdapter, LoRALinear)):
            m.to(device=device, dtype=torch.float16)

    if world_size > 1:
        dist.barrier()

    if rank_id == 0:
        total, trainable = count_params(model)
        log(f"Model: {model.config.model_type}, {n_layers} layers, hidden={h_dim}")
        log(f"LoRA: rank=64, alpha=16, {lora_n} projections")
        log(f"SFM: 4 adapters at layers {sfm_idx}")
        log(f"Params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.2f}%)")

    # ================================================================
    # Phase F: DDP wrap + optimizer
    # ================================================================
    if world_size > 1:
        ddp = nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)
    else:
        ddp = model

    optimizer = torch.optim.AdamW(
        [p for p in ddp.parameters() if p.requires_grad], lr=2e-4)

    if world_size > 1:
        dist.barrier()

    # ================================================================
    # Phase G: Benchmark
    # ================================================================
    if rank_id == 0:
        log("")
        log("=" * 100)
        log("TRAINING THROUGHPUT BENCHMARK")
        log("=" * 100)
        log(f"Configs: {len(CONFIGS)}, Warmup: {WARMUP}, Measure: {MEASURE}")
        log("")

    results = []
    for cfg in CONFIGS:
        results.append(run_one_config(ddp, optimizer, device, cfg,
                                      rank_id, world_size, dist))

    # ================================================================
    # Phase H: Results + upload
    # ================================================================
    if rank_id == 0:
        print_results(results, world_size, output_path)
        log("\nUploading results...")
        try:
            upload_output()
            log("Upload complete.")
        except Exception as e:
            log(f"Upload failed (non-fatal): {e}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank_id == 0:
        log("")
        log("=" * 100)
        log("DONE")
        log("=" * 100)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise
