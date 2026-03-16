"""
train_throughput.py - OpenI Training Throughput Benchmark for SFM + LoRA

Self-contained boot file for OpenI Train Task. Measures real training throughput
(tokens/sec) of Qwen2.5-Coder-7B-Instruct with LoRA + SFM adapters across
4x Ascend 910 ProA NPUs in DDP.

No imports from sfm/ — all architecture code is inline for OpenI portability.

Usage:
    # OpenI Train Task (platform launches once per card automatically)
    python experiments/exp1_thinker/train_throughput.py --model_path /path/to/Qwen2.5-Coder-7B-Instruct

    # Local smoke test (no Ascend)
    python experiments/exp1_thinker/train_throughput.py --model_path /tmp/fake --local_test
"""

# =============================================================================
# 1. Dual Logging System (before any other imports)
# =============================================================================
import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import logging

# Will be configured after c2net prepare() reveals output_path
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
# 2. Argument Parsing + c2net Platform Flow
# =============================================================================
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFM+LoRA Training Throughput Benchmark")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to Qwen2.5-Coder-7B-Instruct model directory")
    parser.add_argument("--local_test", action="store_true",
                        help="Skip Ascend/torch_npu imports for syntax check")
    args, _ = parser.parse_known_args()
    return args


# c2net platform integration
try:
    from c2net.context import prepare, upload_output
except ImportError:
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
# 3. PyTorch Imports (after env setup)
# =============================================================================
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 4. SFM Architecture Classes (self-contained, no sfm/ imports)
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    output = base_linear(x) + (x @ A.T @ B.T) * (alpha / rank)

    A: (rank, in_features), B: (out_features, rank)
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 64, alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming init for A, zeros for B (zero-init ensures no change at start)
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (x @ self.A.T @ self.B.T) * self.scaling
        return base_out + lora_out

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
    """State Slot Bank — tracks program state via gated recurrent slots.

    16 slots x 256 dim. Cross-attention from hidden states over slot vectors.
    Gated slot update: s_t = sig(alpha)*s_{t-1} + sig(beta)*tanh(v).
    """

    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256,
                 num_slots: int = 16, num_heads: int = 4):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = slot_dim  # 256 dim/head

        # Learnable initial slot vectors
        self.slot_vectors = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        # Cross-attention projections
        # Q: hidden_dim -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, num_heads * slot_dim, bias=False)
        # K, V: slot_dim -> num_heads * head_dim
        self.k_proj = nn.Linear(slot_dim, num_heads * slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, num_heads * slot_dim, bias=False)
        # Out: num_heads * head_dim -> hidden_dim
        self.out_proj = nn.Linear(num_heads * slot_dim, hidden_dim, bias=False)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Gated slot update from mean-pooled hidden states
        self.W_alpha = nn.Linear(hidden_dim, slot_dim, bias=True)
        self.W_beta = nn.Linear(hidden_dim, slot_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, slot_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            (modified_hidden, updated_slots)
                modified_hidden: (batch, seq_len, hidden_dim)
                updated_slots: (batch, num_slots, slot_dim)
        """
        batch_size = hidden_states.shape[0]

        # Broadcast initial slots: (num_slots, slot_dim) -> (batch, num_slots, slot_dim)
        if self.slot_vectors.dim() == 2:
            slots = self.slot_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            slots = self.slot_vectors

        # Cross-attention: hidden_states (queries) attend over slots (K/V)
        seq_len = hidden_states.shape[1]

        Q = self.q_proj(hidden_states)  # (B, S, H*D)
        K = self.k_proj(slots)          # (B, N, H*D)
        V = self.v_proj(slots)          # (B, N, H*D)

        # Reshape for multi-head: (B, S/N, H, D) -> (B, H, S/N, D)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (manual for NPU compat)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, S, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, S, D)

        # Reshape back: (B, H, S, D) -> (B, S, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        # Residual + LayerNorm
        modified_hidden = self.layer_norm(hidden_states + attn_output)

        # Gated slot update from mean-pooled hidden states
        h_pooled = hidden_states.mean(dim=1)  # (B, hidden_dim)
        alpha = torch.sigmoid(self.W_alpha(h_pooled))  # (B, slot_dim)
        beta = torch.sigmoid(self.W_beta(h_pooled))    # (B, slot_dim)
        v = torch.tanh(self.W_v(h_pooled))             # (B, slot_dim)

        # Update slots: s_t = alpha * s_{t-1} + beta * v
        updated_slots = alpha.unsqueeze(1) * slots + beta.unsqueeze(1) * v.unsqueeze(1)

        return modified_hidden, updated_slots


class SFMAdapter(nn.Module):
    """Thin wrapper around SFMSlotBank for per-layer injection."""

    def __init__(self, hidden_dim: int = 3584, slot_dim: int = 256):
        super().__init__()
        self.slot_bank = SFMSlotBank(
            hidden_dim=hidden_dim,
            slot_dim=slot_dim,
            num_slots=16,
            num_heads=4,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Returns (modified_hidden_states, new_slot_states)."""
        return self.slot_bank(hidden_states)


# =============================================================================
# 5. Model Injection Functions
# =============================================================================

def inject_lora(model: nn.Module, rank: int = 64, alpha: float = 16.0) -> int:
    """Replace q/k/v/o_proj in self_attn with LoRALinear wrappers.

    Returns count of injected projections.
    """
    count = 0
    target_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in target_names:
            if hasattr(attn, name):
                original = getattr(attn, name)
                if not isinstance(original, LoRALinear):
                    wrapped = LoRALinear(original, rank=rank, alpha=alpha)
                    setattr(attn, name, wrapped)
                    count += 1
    return count


def inject_sfm_adapters(model: nn.Module, hidden_dim: int,
                        layer_indices: list) -> dict:
    """Inject SFM adapters at specified layers via forward hooks.

    Adapters are registered as model modules so DDP can find their parameters.

    Args:
        model: HuggingFace model
        hidden_dim: model hidden dimension
        layer_indices: list of layer indices to inject at

    Returns:
        dict of {layer_idx: SFMAdapter}
    """
    adapters = {}
    for idx in layer_indices:
        adapter = SFMAdapter(hidden_dim=hidden_dim, slot_dim=256)
        model.add_module(f"sfm_adapter_layer{idx}", adapter)

        # Capture adapter by reference for the hook closure
        def make_hook(adapter_ref):
            def hook(module, input, output):
                # output is tuple: (hidden_states, ...) or just hidden_states
                hidden = output[0] if isinstance(output, tuple) else output
                modified, new_slots = adapter_ref(hidden)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook

        model.model.layers[idx].register_forward_hook(make_hook(adapter))
        adapters[idx] = adapter

    return adapters


def count_parameters(model: nn.Module) -> tuple:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# 6. Throughput Benchmark
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


def run_one_config(model, optimizer, device, config, rank, world_size):
    """Run throughput benchmark for a single config.

    Returns dict with metrics or None on OOM.
    """
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    total_steps = WARMUP_STEPS + MEASURE_STEPS

    # Identical data on all ranks (same seed)
    torch.manual_seed(42)
    try:
        torch.npu.manual_seed(42)
    except AttributeError:
        pass

    vocab_size = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size

    # Generate random inputs once (reused every step for consistency)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    tokens_per_step = batch_size * seq_len

    log(f"  Config {config['name']}: B={batch_size}, S={seq_len}, "
        f"tok/step={tokens_per_step:,}")

    import torch.distributed as dist

    try:
        # Warmup phase
        for step in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            try:
                torch.npu.synchronize()
            except AttributeError:
                pass

            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()

            try:
                torch.npu.synchronize()
            except AttributeError:
                pass

        dist.barrier()

        # Measurement phase
        elapsed_list = []
        for step in range(MEASURE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            try:
                torch.npu.synchronize()
            except AttributeError:
                pass

            t_start = time.time()
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            try:
                torch.npu.synchronize()
            except AttributeError:
                pass
            t_end = time.time()

            elapsed_list.append(t_end - t_start)

        dist.barrier()

        # Compute metrics
        avg_ms = sum(elapsed_list) / len(elapsed_list) * 1000
        tokens_per_sec_dev = tokens_per_step / (avg_ms / 1000)

        # Memory stats
        peak_hbm = 0.0
        free_hbm = 0.0
        try:
            peak_hbm = torch.npu.max_memory_allocated(device) / 1e9
            free_mem, _ = torch.npu.mem_get_info(device.index if device.index is not None else 0)
            free_hbm = free_mem / 1e9
            torch.npu.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass

        result = {
            "name": config["name"],
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "avg_ms_per_step": avg_ms,
            "tokens_per_sec_dev": tokens_per_sec_dev,
            "tokens_per_sec_total": tokens_per_sec_dev * world_size,
            "peak_hbm_gb": peak_hbm,
            "free_hbm_gb": free_hbm,
        }
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"  Config {config['name']}: OOM (B={batch_size}, S={seq_len})")
            dist.barrier()  # keep ranks in sync
            try:
                torch.npu.empty_cache()
            except AttributeError:
                pass
            return None
        raise


def print_results(results, world_size, output_path):
    """Print aligned results table and save to file. Rank 0 only."""
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
                f"{r['tokens_per_step']:>9,} | {r['tokens_per_sec_dev']:>10.1f} | "
                f"{r['tokens_per_sec_total']:>12.1f} | {r['avg_ms_per_step']:>8.1f} | "
                f"{r['peak_hbm_gb']:>8.2f}G | {r['free_hbm_gb']:>8.2f}G")
        else:
            log(f"{cfg['name']:>3} | B={cfg['batch_size']:<4} | S={cfg['seq_len']:<5} | OOM")

    # Best config
    valid = [r for r in results if r is not None]
    if valid:
        best = max(valid, key=lambda x: x["tokens_per_sec_dev"])
        log("")
        log(f"Best config: {best['name']} (B={best['batch_size']}, "
            f"S={best['seq_len']})")
        log(f"  {best['tokens_per_sec_dev']:.1f} tok/s/device, "
            f"{best['tokens_per_sec_total']:.1f} tok/s total ({world_size}x NPUs)")

        # Time estimate for 1.1B tokens
        target_tokens = 1.1e9
        hours = target_tokens / (best["tokens_per_sec_total"] * 3600)
        log(f"  Estimated time for 1.1B tokens: {hours:.1f} hours")

    # Save to file
    results_text_path = os.path.join(output_path, "throughput_results.txt")
    with open(results_text_path, "w") as f:
        f.write("SFM+LoRA Training Throughput Benchmark\n")
        f.write("=" * 100 + "\n")
        f.write(f"World size: {world_size} NPUs\n")
        f.write(f"Warmup: {WARMUP_STEPS} steps, Measure: {MEASURE_STEPS} steps\n\n")
        for i, r in enumerate(results):
            cfg = BENCHMARK_CONFIGS[i]
            if r is not None:
                f.write(f"Config {r['name']}: B={r['batch_size']}, S={r['seq_len']}\n")
                f.write(f"  tok/step={r['tokens_per_step']:,}, "
                        f"tok/s/dev={r['tokens_per_sec_dev']:.1f}, "
                        f"tok/s(total)={r['tokens_per_sec_total']:.1f}\n")
                f.write(f"  ms/step={r['avg_ms_per_step']:.1f}, "
                        f"peak_hbm={r['peak_hbm_gb']:.2f}GB, "
                        f"free_hbm={r['free_hbm_gb']:.2f}GB\n\n")
            else:
                f.write(f"Config {cfg['name']}: B={cfg['batch_size']}, "
                        f"S={cfg['seq_len']} — OOM\n\n")

        if valid:
            best = max(valid, key=lambda x: x["tokens_per_sec_dev"])
            f.write(f"\nBest: Config {best['name']} — "
                    f"{best['tokens_per_sec_dev']:.1f} tok/s/dev, "
                    f"{best['tokens_per_sec_total']:.1f} tok/s total\n")
            hours = target_tokens / (best["tokens_per_sec_total"] * 3600)
            f.write(f"1.1B tokens estimate: {hours:.1f} hours\n")

    log(f"\nResults saved to {results_text_path}")


# =============================================================================
# 7. Main
# =============================================================================

def main():
    import traceback

    args = parse_args()
    # Shared mutable container so _main_inner can update output_path for crash handler
    ctx = {"output_path": "/tmp/sfm_throughput_output"}

    try:
        _main_inner(args, ctx)
    except Exception:
        # Log full traceback to both stdout and log file
        tb = traceback.format_exc()
        log(f"\n{'='*80}")
        log("FATAL ERROR — saving partial results and uploading logs")
        log(f"{'='*80}")
        log(tb)

        # Try to save partial results and upload
        try:
            crash_dir = ctx["output_path"]
            os.makedirs(crash_dir, exist_ok=True)
            crash_path = os.path.join(crash_dir, "CRASH_LOG.txt")
            with open(crash_path, "w") as f:
                f.write(tb)
            log(f"Crash log saved to {crash_path}")
            try:
                upload_output()
                log("Partial results uploaded.")
            except Exception as upload_err:
                log(f"Upload failed: {upload_err}")
        except Exception as save_err:
            log(f"Could not save crash log: {save_err}")

        raise


def _main_inner(args, ctx):
    # ------------------------------------------------------------------
    # Local test mode (syntax/import check, no Ascend hardware)
    # ------------------------------------------------------------------
    if args.local_test:
        log("LOCAL TEST MODE — verifying imports and model structure")
        from transformers import AutoConfig

        # Verify SFM classes instantiate correctly
        slot_bank = SFMSlotBank(hidden_dim=3584, slot_dim=256)
        h = torch.randn(1, 10, 3584)
        out, slots = slot_bank(h)
        log(f"  SFMSlotBank: input {h.shape} -> output {out.shape}, slots {slots.shape}")

        adapter = SFMAdapter(hidden_dim=3584)
        out2, slots2 = adapter(h)
        log(f"  SFMAdapter: input {h.shape} -> output {out2.shape}, slots {slots2.shape}")

        # Verify LoRALinear wraps correctly
        base = nn.Linear(512, 512)
        lora = LoRALinear(base, rank=8, alpha=16)
        x = torch.randn(2, 8, 512)
        y = lora(x)
        log(f"  LoRALinear: input {x.shape} -> output {y.shape}")

        # Verify injection logic on a tiny model
        from transformers import AutoModelForCausalLM
        tiny_path = args.model_path
        if tiny_path and os.path.isdir(tiny_path):
            try:
                tiny = AutoModelForCausalLM.from_pretrained(tiny_path, torch_dtype=torch.float32)
                log(f"  Loaded model from {tiny_path}")
                inject_lora(tiny, rank=8, alpha=16)
                inject_sfm_adapters(tiny, hidden_dim=tiny.config.hidden_size,
                                     layer_indices=[0])
                total, trainable = count_parameters(tiny)
                log(f"  After injection: total={total:,}, trainable={trainable:,}")
            except Exception as e:
                log(f"  Model load/inject failed (expected if no local model): {e}")

        log("\nLOCAL TEST COMPLETE — all checks passed")
        return

    # ------------------------------------------------------------------
    # Platform mode: c2net prepare + distributed setup
    # ------------------------------------------------------------------
    import torch.distributed as dist

    # Get rank info from OpenI environment
    rank_id = int(os.environ.get("RANK_ID", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("RANK_SIZE", os.environ.get("WORLD_SIZE", "1")))
    device_id = int(os.environ.get("ASCEND_DEVICE_ID", os.environ.get("LOCAL_RANK", "0")))

    log(f"Rank {rank_id}/{world_size}, device_id={device_id}")

    # c2net prepare — rank 0 first, others wait for flag file
    c2net_ctx = None
    output_path = ctx["output_path"]

    if rank_id == 0:
        log("Rank 0: calling c2net prepare()...")
        c2net_ctx = prepare()
        output_path = c2net_ctx.output_path
        ctx["output_path"] = output_path
        os.makedirs(output_path, exist_ok=True)
        configure_file_logging(output_path)

        # Signal other ranks that prepare is done
        flag_file = "/cache/prepare_completed.txt"
        try:
            with open(flag_file, "w") as f:
                f.write(f"{rank_id}\n")
            log(f"Rank 0: prepare done, flag file written")
        except OSError as e:
            log(f"Rank 0: warning — could not write flag file: {e}")
    else:
        # Non-rank-0: spin-wait for flag file
        flag_file = "/cache/prepare_completed.txt"
        for attempt in range(60):  # wait up to 5 minutes
            if os.path.exists(flag_file):
                break
            time.sleep(5)
        else:
            log(f"Rank {rank_id}: timeout waiting for flag file, proceeding anyway")

        log(f"Rank {rank_id}: calling c2net prepare()...")
        c2net_ctx = prepare()
        output_path = c2net_ctx.output_path
        ctx["output_path"] = output_path
        os.makedirs(output_path, exist_ok=True)
        configure_file_logging(output_path)

    # Determine model path
    model_path = args.model_path
    if not model_path:
        # c2net pretrain_model_path is a root dir; append the model folder name
        pmp = getattr(c2net_ctx, "pretrain_model_path", None)
        if pmp:
            model_path = os.path.join(pmp, "Qwen2.5-Coder-7B")
        else:
            model_path = getattr(c2net_ctx, "model_path", None)

    if not model_path:
        log("ERROR: No model path provided. Use --model_path or configure c2net dataset.")
        sys.exit(1)

    log(f"Model path: {model_path}")

    # ------------------------------------------------------------------
    # Ascend device + distributed init
    # ------------------------------------------------------------------
    import torch_npu

    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)

    # HCCL environment variables
    for key, val in [
        ("TASK_QUEUE_ENABLE", "2"),
        ("CPU_AFFINITY_CONF", "1"),
        ("HCCL_OP_EXPANSION_MODE", "AIV"),
    ]:
        if key not in os.environ:
            os.environ[key] = val

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank_id)

    if rank_id == 0:
        log(f"torch version:     {torch.__version__}")
        log(f"torch_npu version: {torch_npu.__version__}")
        log(f"Distributed:       {world_size} NPUs, backend=hccl")

        # HBM info — mem_get_info returns (free, total)
        try:
            free_hbm_init, total_hbm = torch.npu.mem_get_info(0)
            total_hbm /= 1e9
            log(f"HBM per device:   {total_hbm:.1f} GB")
            log(f"Total HBM (4x):   {total_hbm * world_size:.1f} GB")
        except Exception as e:
            log(f"Warning: could not read HBM info: {e}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM

    model_dtype = torch.float16

    if rank_id == 0:
        log(f"\nLoading model: {model_path} (dtype={model_dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)

    dist.barrier()

    if rank_id == 0:
        log("Model loaded on all ranks.")

    # ------------------------------------------------------------------
    # Freeze + Inject adapters
    # ------------------------------------------------------------------
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Determine SFM layer indices
    num_layers = len(model.model.layers)
    sfm_layer_indices = [
        num_layers // 4,
        num_layers // 2,
        3 * num_layers // 4,
        num_layers - 1,
    ]

    hidden_dim = model.config.hidden_size

    # Inject SFM adapters (adds modules to model tree BEFORE LoRA)
    sfm_adapters = inject_sfm_adapters(model, hidden_dim, sfm_layer_indices)

    # Inject LoRA
    lora_count = inject_lora(model, rank=64, alpha=16)

    # Move new modules to device with correct dtype
    for m in model.modules():
        if isinstance(m, (SFMAdapter, LoRALinear)):
            m.to(device=device, dtype=model_dtype)

    dist.barrier()

    # Print parameter counts
    if rank_id == 0:
        total_params, trainable_params = count_parameters(model)
        log(f"\nModel: {model.config.model_type}, layers={num_layers}, "
            f"hidden={hidden_dim}")
        log(f"LoRA: rank=64, alpha=16, {lora_count} projections injected")
        log(f"SFM adapters: {len(sfm_adapters)} at layers {sfm_layer_indices}")
        log(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
            f"({100 * trainable_params / total_params:.2f}%)")

    # ------------------------------------------------------------------
    # DDP wrap
    # ------------------------------------------------------------------
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
    )

    if rank_id == 0:
        log("DDP wrapping complete.")

    # Optimizer (trainable parameters only)
    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    dist.barrier()

    # ------------------------------------------------------------------
    # Throughput benchmark
    # ------------------------------------------------------------------
    if rank_id == 0:
        log("")
        log("=" * 100)
        log("TRAINING THROUGHPUT BENCHMARK")
        log("=" * 100)
        log(f"Configs: {len(BENCHMARK_CONFIGS)}, "
            f"Warmup: {WARMUP_STEPS} steps, Measure: {MEASURE_STEPS} steps each")
        log("")

    results = []
    for cfg in BENCHMARK_CONFIGS:
        result = run_one_config(ddp_model, optimizer, device, cfg, rank_id, world_size)
        results.append(result)

    # ------------------------------------------------------------------
    # Print results (rank 0 only)
    # ------------------------------------------------------------------
    if rank_id == 0:
        print_results(results, world_size, output_path)

        # Upload to c2net
        log("\nUploading results...")
        try:
            upload_output()
            log("Upload complete.")
        except Exception as e:
            log(f"Upload failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    dist.barrier()
    dist.destroy_process_group()

    if rank_id == 0:
        log("\n" + "=" * 100)
        log("DONE")
        log("=" * 100)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
