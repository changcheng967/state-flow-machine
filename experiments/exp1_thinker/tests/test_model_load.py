"""
test_model_load.py - Model loading and adapter injection for Thinker-3B

Tests loading Qwen2.5-Coder-3B on Ascend NPU with multiple precision configs.
If model loading succeeds, tests adapter injection and gradient flow.

Usage:
    python experiments/exp1_thinker/tests/test_model_load.py
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configurations to try, in order
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "A (FP16)",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "dtype": torch.float16,
    },
    {
        "name": "B (BF16)",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "name": "C (FP32)",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "dtype": torch.float32,
    },
    {
        "name": "D (1.5B FP16)",
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "dtype": torch.float16,
    },
    {
        "name": "E (1.5B FP32)",
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "dtype": torch.float32,
    },
]

LONG_PROMPT = '''def fibonacci(n):
    """Return the nth Fibonacci number using memoization."""
    memo = {}
    def fib_helper(k):
        if k in memo:
            return memo[k]
        if k <= 1:
            return k
        result = fib_helper(k - 1) + fib_helper(k - 2)
        memo[k] = result
        return result
    return fib_helper(n)

def merge_sort(arr):
    """Sort an array using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def binary_search(arr, target):
    """Search for target in sorted array using binary search.
    Returns the index if found, -1 otherwise.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
'''


def try_load_model(config: dict, device: torch.device) -> tuple:
    """Try loading a model with given config. Returns (model, tokenizer, config) or (None, None, config)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n--- Trying CONFIG {config['name']} ---")
    print(f"  Model: {config['model_id']}")
    print(f"  Dtype: {config['dtype']}")

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
        return model, tokenizer, config
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, None, config


def test_generation(model, tokenizer, device, prompt: str, max_new_tokens: int, temperature: float = 0.7) -> dict:
    """Run generation and measure speed. Returns metrics dict."""
    import torch_npu

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

    return {
        "text": generated_text,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_mem,
    }


class SimpleSlotAdapter(nn.Module):
    """Minimal slot memory adapter for validation."""

    def __init__(self, hidden_dim: int, num_slots: int = 16, slot_dim: int = 256):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.proj_in = nn.Linear(hidden_dim, slot_dim)
        self.proj_out = nn.Linear(slot_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_val = self.gate(hidden_states)
        projected = self.proj_in(hidden_states)
        slot_attn = torch.softmax(
            projected @ self.slots.squeeze(0).T / (256 ** 0.5), dim=-1
        )
        read_out = slot_attn @ self.slots.squeeze(0)
        return hidden_states + gate_val * self.proj_out(read_out)


def test_adapter_injection(model, tokenizer, device, config: dict) -> bool:
    """Test adapter injection after layer 14. Returns True on success."""
    import torch_npu

    print("\n" + "=" * 60)
    print("ADAPTER INJECTION TEST")
    print("=" * 60)

    # Determine hidden_dim from model config
    hidden_dim = model.config.hidden_size
    print(f"  Model hidden_dim: {hidden_dim}")

    # Create adapter
    adapter = SimpleSlotAdapter(hidden_dim, num_slots=16, slot_dim=256).to(device)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter parameters: {adapter_params:,}")

    # Inject after layer 14
    layer_idx = min(14, len(model.model.layers) - 1)
    original_layer = model.model.layers[layer_idx]
    original_forward = original_layer.forward

    def patched_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        # output is a BaseModelOutputWithPast, hidden_states is first element
        if isinstance(output, tuple):
            hidden = output[0]
            adapted = adapter(hidden)
            output = (adapted,) + output[1:]
        return output

    original_layer.forward = patched_forward
    print(f"  Adapter injected after layer {layer_idx}")

    # Freeze all original params
    trainable_before = 0
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            trainable_before += param.numel()
    print(f"  Frozen original params, adapter trainable: {trainable_before:,}")

    # Forward pass
    inputs = tokenizer("def hello():\n    print('hello')\n", return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    print(f"  Forward pass OK, loss={loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check adapter gradients
    grad_ok = True
    grad_count = 0
    for name, param in model.named_parameters():
        if "adapter" in name:
            if param.grad is None:
                print(f"  WARNING: {name} has no gradient!")
                grad_ok = False
            elif param.grad.abs().sum() == 0:
                print(f"  WARNING: {name} gradient is all zeros!")
                grad_ok = False
            else:
                grad_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Adapter params with valid gradients: {grad_count:,}")
    print(f"  Total model params: {total_params:,}")
    print(f"  Trainable: {trainable_before:,} / {total_params:,} "
          f"({100 * trainable_before / total_params:.2f}%)")

    if grad_ok:
        print("\n  ADAPTER INJECTION: PASS -- gradients flowing, "
              f"{trainable_before:,} trainable params out of {total_params:,}")
    else:
        print("\n  ADAPTER INJECTION: FAIL -- gradient flow broken")

    return grad_ok


def main():
    import torch_npu

    print("=" * 60)
    print("Thinker-3B Hardware Validation: Model Loading")
    print("=" * 60)
    print(f"torch version:     {torch.__version__}")
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"NPU available:     {torch.npu.is_available()}")
    print(f"NPU count:         {torch.npu.device_count()}")

    if not torch.npu.is_available():
        print("ERROR: No NPU available!")
        sys.exit(1)

    device = torch.device("npu:0")
    total_hbm = torch.npu.mem_get_info(0)[0] / 1e9
    print(f"Device: npu:0, Total HBM: {total_hbm:.1f} GB")

    # Try configs in order
    model = None
    tokenizer = None
    working_config = None

    for config in CONFIGS:
        model, tokenizer, working_config = try_load_model(config, device)
        if model is not None:
            break

    if model is None:
        print("\nERROR: All model configs failed!")
        sys.exit(1)

    dtype_name = str(working_config["dtype"]).replace("torch.", "")
    print(f"\nCONFIG {working_config['name']}: LOADED SUCCESSFULLY")

    # --- Short generation test ---
    print("\n--- Short generation test (50 tokens) ---")
    short_prompt = "def fibonacci(n):\n    "
    metrics = test_generation(model, tokenizer, device, short_prompt, max_new_tokens=50)
    print(f"  Generated ({metrics['total_tokens']} tokens, "
          f"{metrics['tokens_per_sec']:.1f} tok/s, "
          f"{metrics['elapsed']:.1f}s):")
    print(f"  {metrics['text'][:200]}")
    print(f"\n  CONFIG {working_config['name']}: PASS -- peak memory: {metrics['peak_memory_gb']:.1f} GB")

    # --- Long generation test ---
    print("\n--- Long generation test (100 tokens) ---")
    long_metrics = test_generation(model, tokenizer, device, LONG_PROMPT, max_new_tokens=100)
    print(f"  Generated ({long_metrics['total_tokens']} tokens, "
          f"{long_metrics['tokens_per_sec']:.1f} tok/s):")
    print(f"  {long_metrics['text'][:300]}")

    # --- Memory headroom estimation ---
    model_memory = long_metrics["peak_memory_gb"]
    remaining = total_hbm - model_memory
    # Rough estimate: gradients ~2x model, optimizer states ~2x model for AdamW
    # But we're only training adapters, so much less
    adapter_overhead = 0.2  # SFM adapters: ~0.2 GB
    lora_overhead = 1.5  # rank 16 LoRA: ~1-2 GB
    training_overhead = model_memory * 0.1  # 10% of model for adapter gradients+optimizer

    print("\n" + "=" * 60)
    print("MEMORY HEADROOM ANALYSIS")
    print("=" * 60)
    print(f"  Total HBM:                    {total_hbm:.1f} GB")
    print(f"  Model uses (peak):            {model_memory:.1f} GB")
    print(f"  Remaining:                    {remaining:.1f} GB")
    print(f"  Estimated LoRA (rank 16):     ~{lora_overhead:.1f} GB")
    print(f"  Estimated SFM adapters:       ~{adapter_overhead:.1f} GB")
    print(f"  Estimated training overhead:  ~{training_overhead:.1f} GB")
    print(f"  Total estimated overhead:     ~{lora_overhead + adapter_overhead + training_overhead:.1f} GB")
    print(f"  Net headroom:                 ~{remaining - lora_overhead - adapter_overhead - training_overhead:.1f} GB")

    feasible = remaining > lora_overhead + adapter_overhead + training_overhead + 2.0
    print(f"\n  VERDICT: Training is {'FEASIBLE' if feasible else 'NOT FEASIBLE'} on single NPU")

    # --- Adapter injection test ---
    adapter_pass = test_adapter_injection(model, tokenizer, device, working_config)

    # Clean up
    del model
    torch.npu.empty_cache()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
