"""
train_throughput.py - OpenI Training Throughput Benchmark for SFM + LoRA

Self-contained boot file for OpenI Train Task. Measures real training throughput
(tokens/sec) of Qwen2.5-Coder-7B-Instruct with LoRA + SFM adapters across
4x Ascend 910 ProA NPUs in DDP.

Two modes:
  Boot mode  (OpenI platform calls this once per card):
    - Rank 0: c2net prepare() -> launch torchrun subprocess
    - Other ranks: exit immediately (following OpenI_LLM_Finetune_Example pattern)
  Worker mode (torchrun launches this for each NPU):
    - Standard PyTorch DDP with ASCEND_RT_VISIBLE_DEVICES per worker

No imports from sfm/ — all architecture code is inline for OpenI portability.
"""

import os
import sys
import time
import argparse
import subprocess
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# c2net (module-level import, prepare() called in boot mode only)
# =============================================================================
try:
    from c2net.context import prepare, upload_output
except ImportError:
    def prepare():
        class _DummyCtx:
            output_path = "/tmp/sfm_throughput_output"
            pretrain_model_path = None
            dataset_path = None
        return _DummyCtx()
    def upload_output():
        pass


# =============================================================================
# Logging
# =============================================================================
_logger_configured = False


def log(msg: str) -> None:
    print(msg, flush=True)
    if _logger_configured:
        import logging
        logging.info(msg)


def configure_logging(output_path: str) -> None:
    global _logger_configured
    import logging
    log_path = os.path.join(output_path, "training.log")
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", filemode="w",
    )
    _logger_configured = True
    log(f"File logging: {log_path}")


# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser(description="SFM+LoRA Throughput Benchmark")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--local_test", action="store_true")


# =============================================================================
# SFM Architecture Classes (self-contained)
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper: output = base(x) + (x @ A.T @ B.T) * scale."""

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


# =============================================================================
# Injection
# =============================================================================

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

        def _hook(adapter_ref):
            def hook(module, inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                modified, _ = adapter_ref(hidden)
                return (modified,) + output[1:] if isinstance(output, tuple) else modified
            return hook

        model.model.layers[idx].register_forward_hook(_hook(adapter))
        adapters[idx] = adapter
    return adapters


def count_params(model: nn.Module) -> tuple:
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, tr


# =============================================================================
# Benchmark configs
# =============================================================================

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


# =============================================================================
# Worker mode (launched by torchrun)
# =============================================================================

def _sync():
    try:
        torch.npu.synchronize()
    except AttributeError:
        pass


def run_one_config(model, optimizer, device, cfg, rank, world_size, dist):
    B, S = cfg["batch_size"], cfg["seq_len"]
    tok_per_step = B * S

    torch.manual_seed(42)
    try:
        torch.npu.manual_seed(42)
    except AttributeError:
        pass

    vocab = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size
    input_ids = torch.randint(0, vocab, (B, S), device=device)
    labels = input_ids.clone()

    log(f"  Config {cfg['name']}: B={B}, S={S}, tok/step={tok_per_step:,}")

    try:
        for _ in range(WARMUP):
            optimizer.zero_grad(set_to_none=True)
            _sync()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            _sync()
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
        dist.barrier()

        avg_ms = sum(elapsed) / len(elapsed) * 1000
        tps = tok_per_step / (avg_ms / 1000)

        peak = free = 0.0
        try:
            peak = torch.npu.max_memory_allocated(device) / 1e9
            free = torch.npu.mem_get_info(device.index if device.index is not None else 0)[0] / 1e9
            torch.npu.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass

        return {"name": cfg["name"], "batch_size": B, "seq_len": S,
                "tok_per_step": tok_per_step, "avg_ms": avg_ms,
                "tps_dev": tps, "tps_total": tps * world_size,
                "peak_hbm": peak, "free_hbm": free}

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"  Config {cfg['name']}: OOM (B={B}, S={S})")
            dist.barrier()
            try:
                torch.npu.empty_cache()
            except AttributeError:
                pass
            return None
        raise


def run_worker(args):
    """torchrun worker: standard PyTorch DDP across NPUs."""
    import torch.distributed as dist
    import torch_npu

    # Ascend env vars BEFORE torch_npu operations
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
    os.environ.setdefault("CPU_AFFINITY_CONF", "1")
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Device isolation: each worker uses only its own NPU
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(local_rank)

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    dist.init_process_group(backend="hccl")

    if rank == 0 and args.output_path:
        configure_logging(args.output_path)
        log(f"Worker: rank={rank}/{world_size}, local_rank={local_rank}")
        log(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}")

        try:
            free_i, total_i = torch.npu.mem_get_info(0)
            log(f"HBM: {total_i/1e9:.1f} GB total, {free_i/1e9:.1f} GB free")
        except Exception as e:
            log(f"HBM info failed: {e}")

    # Load model
    model_path = args.model_path
    if rank == 0:
        log(f"Loading model: {model_path}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
    )
    model = model.to(device)
    dist.barrier()

    if rank == 0:
        log("Model loaded on all ranks.")

    # Freeze + inject
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
    dist.barrier()

    if rank == 0:
        total, trainable = count_params(model)
        log(f"Model: {model.config.model_type}, {n_layers} layers, hidden={h_dim}")
        log(f"LoRA: rank=64, alpha=16, {lora_n} projections")
        log(f"SFM: 4 adapters at layers {sfm_idx}")
        log(f"Params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.2f}%)")

    # DDP
    ddp = nn.parallel.DistributedDataParallel(
        model, device_ids=[0], output_device=0)
    optimizer = torch.optim.AdamW(
        [p for p in ddp.parameters() if p.requires_grad], lr=2e-4)
    dist.barrier()

    # Benchmark
    if rank == 0:
        log("")
        log("=" * 100)
        log("TRAINING THROUGHPUT BENCHMARK")
        log("=" * 100)
        log(f"Configs: {len(CONFIGS)}, Warmup: {WARMUP}, Measure: {MEASURE}")
        log("")

    results = []
    for cfg in CONFIGS:
        results.append(run_one_config(ddp, optimizer, device, cfg,
                                      rank, world_size, dist))

    # Results (rank 0 only)
    if rank == 0 and args.output_path:
        _print_results(results, world_size, args.output_path)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        log("")
        log("=" * 100)
        log("DONE")
        log("=" * 100)


def _print_results(results, world_size, output_path):
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
                f"{r['tok_per_step']:>9,} | {r['tps_dev']:>10.1f} | "
                f"{r['tps_total']:>12.1f} | {r['avg_ms']:>8.1f} | "
                f"{r['peak_hbm']:>8.2f}G | {r['free_hbm']:>8.2f}G")
        else:
            log(f"{cfg['name']:>3} | B={cfg['batch_size']:<4} | "
                f"S={cfg['seq_len']:<5} | OOM")

    valid = [r for r in results if r is not None]
    if valid:
        best = max(valid, key=lambda x: x["tps_dev"])
        log(f"\nBest: {best['name']} (B={best['batch_size']}, "
            f"S={best['seq_len']})")
        log(f"  {best['tps_dev']:.1f} tok/s/device, "
            f"{best['tps_total']:.1f} tok/s total ({world_size}x NPUs)")
        hours = 1.1e9 / (best["tps_total"] * 3600)
        log(f"  1.1B tokens estimate: {hours:.1f} hours")

    # Save file
    path = os.path.join(output_path, "throughput_results.txt")
    with open(path, "w") as f:
        f.write("SFM+LoRA Training Throughput Benchmark\n")
        f.write(f"World size: {world_size} NPUs\n\n")
        for i, r in enumerate(results):
            cfg = CONFIGS[i]
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
    log(f"Results saved to {path}")


# =============================================================================
# Boot mode (OpenI platform entry point, rank 0 only)
# =============================================================================

def resolve_model_path(c2net_ctx, args):
    """Resolve model path from args or c2net context with auto-detection."""
    if args.model_path:
        return args.model_path

    pmp = getattr(c2net_ctx, "pretrain_model_path", None)
    if pmp and os.path.isdir(pmp):
        try:
            entries = [e for e in os.listdir(pmp)
                       if os.path.isdir(os.path.join(pmp, e))]
            log(f"Model root: {pmp} — subdirs: {entries}")
            for name in ["Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B"]:
                if name in entries:
                    return os.path.join(pmp, name)
            if len(entries) == 1:
                return os.path.join(pmp, entries[0])
            if os.path.exists(os.path.join(pmp, "config.json")):
                return pmp
        except OSError as e:
            log(f"Could not list model dir: {e}")

    mp = getattr(c2net_ctx, "model_path", None)
    return mp


def run_boot(args):
    """OpenI boot: prepare() -> detect NPUs -> launch torchrun subprocess."""
    c2net_ctx = prepare()
    output_path = c2net_ctx.output_path
    os.makedirs(output_path, exist_ok=True)
    configure_logging(output_path)

    model_path = resolve_model_path(c2net_ctx, args)
    if not model_path:
        log("ERROR: no model path found. Upload a model dataset on OpenI.")
        try:
            upload_output()
        except Exception:
            pass
        sys.exit(1)

    log(f"Model path: {model_path}")

    # Detect NPUs (rank 0 sees all of them on OpenI)
    try:
        import torch_npu
        npu_count = torch.npu.device_count()
    except ImportError:
        log("ERROR: torch_npu not available")
        sys.exit(1)

    log(f"NPU count: {npu_count}")

    if npu_count == 0:
        log("ERROR: no NPUs detected")
        sys.exit(1)

    # Build torchrun command (following OpenI_LLM_Finetune_Example pattern)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(npu_count))
    env.setdefault("TASK_QUEUE_ENABLE", "2")
    env.setdefault("CPU_AFFINITY_CONF", "1")
    env.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(npu_count),
        __file__,
        "--_worker",
        "--model_path", model_path,
        "--output_path", output_path,
    ]

    log(f"Launching: {' '.join(cmd)}")

    # Stream output (same pattern as OpenI_LLM_Finetune_Example)
    p = subprocess.Popen(cmd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(p.stdout.readline, b''):
        text = line.decode('utf-8', errors='replace').strip()
        if text:
            log(text)
    p.wait()

    if p.returncode != 0:
        log(f"torchrun exited with code {p.returncode}")
    else:
        log("torchrun completed successfully.")

    # Upload results
    log("Uploading results...")
    try:
        upload_output()
        log("Upload complete.")
    except Exception as e:
        log(f"Upload failed (non-fatal): {e}")


# =============================================================================
# Local test mode
# =============================================================================

def run_local_test():
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


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    if args._worker:
        # torchrun worker mode — run actual DDP benchmark
        run_worker(args)
    elif args.local_test:
        # Local syntax/import check
        run_local_test()
    else:
        # Boot mode — OpenI platform entry point
        rank_id = os.getenv('RANK_ID')
        if rank_id is not None and int(rank_id) != 0:
            # Multi-card mode: only rank 0 does work
            # (following OpenI_LLM_Finetune_Example pattern)
            sys.exit(0)

        try:
            run_boot(args)
        except Exception:
            tb = traceback.format_exc()
            log(f"\nFATAL ERROR:\n{tb}")
            raise
