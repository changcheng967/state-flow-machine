"""
test_model_load.py - Distributed model loading for Thinker (DDP across 4 NPUs)

Tests loading Qwen2.5-Coder-7B-Instruct in DDP across all 4 Ascend NPUs.
Tries FP16, BF16, FP32 in order. Falls back to 3B if all 7B configs fail.

Usage:
    torchrun --nproc_per_node=4 experiments/exp1_thinker/tests/test_model_load.py
"""

import os
import sys
import time

# CRITICAL: HF mirror + ASCEND device isolation BEFORE any imports
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if is_torchrun and "LOCAL_RANK" in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    os.environ["TASK_QUEUE_ENABLE"] = "2"
    os.environ["CPU_AFFINITY_CONF"] = "1"
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    os.environ["HCCL_DETERMINISTIC"] = "true"

import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Configurations to try, in order
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "A (7B FP16)",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "dtype": torch.float16,
    },
    {
        "name": "B (7B BF16)",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "name": "C (7B FP32)",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "dtype": torch.float32,
    },
    {
        "name": "D (3B FP16 fallback)",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "dtype": torch.float16,
    },
]

SHORT_PROMPT = "def fibonacci(n):\n    "


def try_load_ddp(config: dict, rank: int, world_size: int) -> tuple:
    """Try loading a model in DDP. Returns (model, tokenizer, config) or (None, None, config)."""
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    if rank == 0:
        print(f"\n--- Trying CONFIG {config['name']} ---")
        print(f"  Model: {config['model_id']}")
        print(f"  Dtype: {config['dtype']}")
        print(f"  World size: {world_size}")
        print(f"  Loading model on rank 0...", flush=True)

    dist.barrier()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=config["dtype"],
            trust_remote_code=True,
        )
        model = model.to(device)

        # Sync all ranks have model loaded
        dist.barrier()

        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[0], output_device=0
        )

        total_params = sum(p.numel() for p in ddp_model.parameters())
        if rank == 0:
            print(f"  Model loaded on all ranks. Total params: {total_params:,}")

        return ddp_model, tokenizer, config
    except Exception as e:
        dist.barrier()
        if rank == 0:
            print(f"  FAILED: {e}")
        return None, None, config


def test_generation(model, tokenizer, device, prompt: str,
                   max_new_tokens: int, temperature: float = 0.7) -> dict:
    """Run generation on rank 0. Returns metrics dict (empty on other ranks)."""
    import torch_npu

    if dist.get_rank() != 0:
        dist.barrier()
        return {}

    torch.npu.reset_peak_memory_stats(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_tokens = generated_ids.shape[0]
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    peak_mem = torch.npu.max_memory_allocated(device) / 1e9
    _, free_mem = torch.npu.mem_get_info(0)
    free_mem_gb = free_mem / 1e9

    dist.barrier()  # sync after generation

    return {
        "text": generated_text,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_mem,
        "free_memory_gb": free_mem_gb,
    }


def test_training_step(model, tokenizer, rank: int, world_size: int, device: torch.device):
    """Single forward+backward pass to confirm gradients flow through DDP."""
    import torch_npu

    if rank == 0:
        print("\n--- Training step test (forward + backward) ---")

    torch.npu.reset_peak_memory_stats(device)

    # Create dummy batch (same on all ranks for consistency)
    torch.manual_seed(42)
    batch_size = 4
    inputs = tokenizer(
        "def add(a, b):\n    return a + b\n\nprint(add(1, 2))",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    # Forward
    try:
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    except Exception as e:
        if rank == 0:
            print(f"  Forward FAIL: {e}")
        dist.barrier()
        return False

    # Backward
    try:
        loss.backward()
    except Exception as e:
        if rank == 0:
            print(f"  Backward FAIL: {e}")
        dist.barrier()
        return False

    # Verify gradients exist on all ranks
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_p = sum(1 for p in model.parameters())

    if rank == 0:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients: {has_grad}/{total_p} params have non-None grads")
        print(f"  Training step: PASS")

    dist.barrier()
    return True


def main():
    import torch_npu

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print("=" * 60)
        print("Thinker Hardware Validation: Distributed Model Loading")
        print("=" * 60)
        print(f"torch version:     {torch.__version__}")
        print(f"torch_npu version: {torch_npu.__version__}")
        print(f"World size:       {world_size}")

    device = torch.device("npu:0")

    # Initialize process group
    dist.init_process_group(backend="hccl")
    dist.barrier()

    total_hbm = torch.npu.mem_get_info(0)[0] / 1e9
    if rank == 0:
        print(f"HBM per device:   {total_hbm:.1f} GB")
        print(f"Total HBM (4x):   {total_hbm * world_size:.1f} GB")

    # Try configs in order
    model = None
    tokenizer = None
    working_config = None

    for config in CONFIGS:
        # Destroy previous DDP model if any
        if model is not None:
            del model
            if "npu" in str(device):
                torch.npu.empty_cache()

        model, tokenizer, working_config = try_load_ddp(config, rank, world_size)
        if model is not None:
            break

    if model is None:
        if rank == 0:
            print("\nERROR: All model configs failed!")
        dist.destroy_process_group()
        sys.exit(1)

    # --- Generation test (rank 0 only) ---
    metrics = test_generation(model, tokenizer, device, SHORT_PROMPT, max_new_tokens=50)

    # --- Training step test (all ranks) ---
    train_ok = test_training_step(model, tokenizer, rank, world_size, device)

    # --- Summary (rank 0 only) ---
    if rank == 0:
        cfg = working_config
        dtype_name = str(cfg["dtype"]).replace("torch.", "")

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Model:       {cfg['model_id']}")
        print(f"  Precision:  {dtype_name}")
        print(f"  Config:      {cfg['name']}")
        print(f"  World size:  {world_size} NPUs")
        print(f"  Peak memory: {metrics['peak_memory_gb']:.1f} GB per device")
        print(f"  Free memory: {metrics['free_memory_gb']:.1f} GB per device")
        print(f"  Generation:  {metrics['tokens_per_sec']:.1f} tokens/sec")
        print(f"  Training:    {'PASS' if train_ok else 'FAIL'}")

        # Memory headroom for training
        model_mem = metrics["peak_memory_gb"]
        free = metrics["free_memory_gb"]
        # DDP model params: gradients ~1x model, optimizer (AdamW) ~2x model
        # But for LoRA/adapter-only training: gradients + optimizer ~0.5x model
        lora_overhead = 1.5   # rank 16 LoRA
        sfm_overhead = 0.2    # SFM slot adapters
        grad_opt = model_mem * 0.5  # adapter gradients + optimizer states
        activation_overhead = 2.0  # activations + KV cache
        total_overhead = lora_overhead + sfm_overhead + grad_opt + activation_overhead
        net_headroom = free - total_overhead

        print(f"\n  Memory per device ({total_hbm:.1f} GB total):")
        print(f"    Model:         {model_mem:.1f} GB")
        print(f"    Remaining:     {free:.1f} GB")
        print(f"    LoRA (r=16):   ~{lora_overhead:.1f} GB")
        print(f"    SFM adapters:  ~{sfm_overhead:.1f} GB")
        print(f"    Grad+Optimizer:~{grad_opt:.1f} GB")
        print(f"    Activations:   ~{activation_overhead:.1f} GB")
        print(f"    Total overhead:~{total_overhead:.1f} GB")
        print(f"    Net headroom:  ~{net_headroom:.1f} GB")

        # Estimate max batch size (rough: activations scale linearly)
        if net_headroom > 4.0:
            est_batch = max(1, int(net_headroom / 2.0))
        else:
            est_batch = 1

        print(f"\n  Estimated max batch size: ~{est_batch}")
        print(f"  Recommended: batch_size={min(est_batch, 4)}, "
              f"gradient_accumulation={max(1, 8 // min(est_batch, 4))}")

        feasible = net_headroom > 2.0 and train_ok
        print(f"\n  VERDICT: {'PASS' if feasible else 'FAIL'} — "
              f"Thinker with {cfg['model_id']} ({dtype_name}, "
              f"{world_size}x NPU DDP) is "
              f"{'FEASIBLE' if feasible else 'NOT FEASIBLE'}")

        # Write results to file
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "model_load_results.txt"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            f.write(f"Thinker Hardware Validation: Distributed Model Loading\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Model:       {cfg['model_id']}\n")
            f.write(f"Precision:  {dtype_name}\n")
            f.write(f"Config:      {cfg['name']}\n")
            f.write(f"World size:  {world_size} NPUs\n")
            f.write(f"Peak memory: {metrics['peak_memory_gb']:.1f} GB per device\n")
            f.write(f"Free memory: {metrics['free_memory_gb']:.1f} GB per device\n")
            f.write(f"Generation:  {metrics['tokens_per_sec']:.1f} tokens/sec\n")
            f.write(f"Training:    {'PASS' if train_ok else 'FAIL'}\n\n")
            f.write(f"Generated text (50 tokens):\n{metrics['text']}\n\n")
            f.write(f"VERDICT: {'PASS' if feasible else 'FAIL'}\n")
        print(f"\n  Results saved to {results_path}")

    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)


if __name__ == "__main__":
    main()
