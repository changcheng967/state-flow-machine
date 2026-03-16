"""
test_hardware.py - Single NPU validation for Thinker-3B

Validates each Ascend 910 ProA NPU individually:
- Device detection and properties
- HBM memory query (total and free)
- 1GB tensor allocation, matmul, verify, free

Usage:
    python experiments/exp1_thinker/tests/test_hardware.py
"""

import torch
import torch_npu
import time
import sys


def test_single_npu(device_id: int) -> dict:
    """Test a single NPU device. Returns result dict."""
    device = torch.device(f"npu:{device_id}")
    result = {"device_id": device_id, "pass": False, "details": []}

    try:
        # Device properties
        props = torch.npu.get_device_properties(device_id)
        result["details"].append(f"  Name: {props.name}")
        result["details"].append(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
        result["details"].append(f"  Major: {props.major}, Minor: {props.minor}")

        # Memory query
        total_mem, free_mem = torch.npu.mem_get_info(device_id)
        result["details"].append(f"  HBM total: {total_mem / 1e9:.1f} GB")
        result["details"].append(f"  HBM free:  {free_mem / 1e9:.1f} GB")
        result["total_hbm"] = total_mem
        result["free_hbm"] = free_mem

        # Allocate 1GB tensor
        gb_tensor = torch.randn(1024, 1024, 256, dtype=torch.float16, device=device)  # 0.5 GB
        result["details"].append(f"  Allocated 0.5 GB tensor: shape {gb_tensor.shape}")

        # Matmul test
        a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
        b = torch.randn(1024, 1024, dtype=torch.float16, device=device)
        c = torch.matmul(a, b)
        expected_sum = c.sum().item()

        # Verify result is finite
        assert torch.isfinite(c).all(), "Matmul produced non-finite values"
        result["details"].append(f"  Matmul (1024x1024): sum={expected_sum:.1f} (finite)")

        # Free tensors
        del gb_tensor, a, b, c
        torch.npu.empty_cache()

        after_free = torch.npu.mem_get_info(device_id)[1]
        result["details"].append(f"  After free: {after_free / 1e9:.1f} GB free")

        result["pass"] = True
    except Exception as e:
        result["details"].append(f"  ERROR: {e}")

    return result


def main():
    print("=" * 60)
    print("Thinker-3B Hardware Validation: Single NPU")
    print("=" * 60)

    # Versions
    print(f"\ntorch version:      {torch.__version__}")
    print(f"torch_npu version:  {torch_npu.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    print(f"NPU available:      {torch.npu.is_available()}")
    print(f"NPU device count:   {torch.npu.device_count()}")

    n_devices = torch.npu.device_count()
    if n_devices == 0:
        print("\nERROR: No NPU devices detected!")
        sys.exit(1)

    print(f"\n--- Testing {n_devices} NPU device(s) ---\n")

    results = []
    total_hbm = 0

    for i in range(n_devices):
        print(f"[npu:{i}]")
        r = test_single_npu(i)
        results.append(r)
        if r["pass"]:
            total_hbm += r.get("total_hbm", 0)
            print(f"npu:{i} -- PASS")
        else:
            print(f"npu:{i} -- FAIL")
        for line in r["details"]:
            print(line)
        print()

    # Summary
    passed = sum(1 for r in results if r["pass"])
    print("=" * 60)
    print(f"SUMMARY: {passed}/{n_devices} NPUs passed")
    print(f"Total HBM across all cards: {total_hbm / 1e9:.1f} GB")
    print("=" * 60)

    return passed == n_devices


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
