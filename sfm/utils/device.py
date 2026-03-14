"""
Device selection for SFM - Ascend NPU only.

This architecture is designed for Huawei Ascend NPUs.
torch_npu is required.

NPU OPTIMIZATION:
- Disable JIT recompilation for stable performance
- Set HCCL environment variables for 4-NPU distributed training
- Proper memory management and synchronization
"""

import os
import torch
from typing import Tuple

# Set HCCL environment variables BEFORE importing torch_npu
os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '1200')
os.environ.setdefault('HCCL_EXEC_TIMEOUT', '1200')


def get_device() -> torch.device:
    """
    Get Ascend NPU device.

    Returns:
        torch.device: Ascend NPU device.

    Raises:
        RuntimeError: If torch_npu is not available.
    """
    try:
        import torch_npu

        # Disable JIT recompilation for stable performance
        try:
            torch.npu.set_compile_mode(jit_compile=False)
        except AttributeError:
            pass  # Older versions may not have this

        if torch.npu.is_available():
            device = torch.device("npu:0")
            print(f"[Device] Using Ascend NPU: {torch.npu.get_device_name(0)}")
            return device
        else:
            raise RuntimeError("Ascend NPU is not available")
    except ImportError:
        raise RuntimeError(
            "torch_npu is required for State-Flow Machine.\n"
            "Install it with: pip install torch-npu\n"
            "This architecture is designed for Huawei Ascend NPUs."
        )


def get_device_info() -> Tuple[str, int, str]:
    """
    Get Ascend NPU device information.

    Returns:
        Tuple of (device_type, memory_gb, device_name)
    """
    import torch_npu

    memory_gb = torch.npu.get_device_properties(0).total_memory / (1024**3)
    name = torch.npu.get_device_name(0)
    return ("npu", int(memory_gb), name)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    import torch_npu
    torch.npu.manual_seed_all(seed)


def to_device(data, device: torch.device):
    """
    Recursively move data to device.

    Handles tensors, dicts, lists, and tuples.

    Args:
        data: Data to move.
        device: Target device.

    Returns:
        Data on the target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def synchronize():
    """Synchronize NPU device (useful for timing measurements)."""
    try:
        import torch_npu
        torch.npu.synchronize()
    except (ImportError, AttributeError):
        pass


def setup_npu_optimizations():
    """
    Apply NPU-specific optimizations.

    Call this once at the start of training for best performance.
    """
    import torch_npu

    # Disable JIT recompilation for stable performance
    try:
        torch.npu.set_compile_mode(jit_compile=False)
        print("[NPU] Disabled JIT recompilation for stable performance")
    except AttributeError:
        pass

    # Set memory configuration
    try:
        # Enable memory optimization
        torch.npu.set_memory_fraction(0.9, 0)  # Use 90% of memory
        print("[NPU] Memory optimization enabled")
    except AttributeError:
        pass

    # Print device info
    try:
        name = torch.npu.get_device_name(0)
        memory = torch.npu.get_device_properties(0).total_memory / (1024**3)
        print(f"[NPU] Device: {name}, Memory: {memory:.1f} GB")
    except (AttributeError, IndexError):
        pass


def get_npu_count() -> int:
    """Get the number of available NPUs."""
    try:
        import torch_npu
        return torch.npu.device_count()
    except (ImportError, AttributeError):
        return 0


def print_distributed_info():
    """Print distributed training information."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            print(f"[Distributed] Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
            print(f"[Distributed] Backend: {dist.get_backend()}")
        else:
            print("[Distributed] Not initialized")
    except ImportError:
        print("[Distributed] torch.distributed not available")


if __name__ == "__main__":
    # Smoke test
    print("=" * 50)
    print("SFM Device Selection Test - Ascend NPU Only")
    print("=" * 50)

    try:
        device = get_device()
        device_type, memory_gb, name = get_device_info()

        print(f"\nSelected Device: {device}")
        print(f"Device Type: {device_type}")
        print(f"Device Name: {name}")
        print(f"Available Memory: {memory_gb} GB")

        # Test NPU optimizations
        print("\nTesting NPU optimizations...")
        setup_npu_optimizations()

        # Test tensor creation on device
        print("\nTesting tensor operations...")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        synchronize()  # For accurate timing
        print(f"Matrix multiplication successful: result shape = {z.shape}")

        # Test seed setting
        set_seed(42)
        a = torch.randn(10, device=device)
        set_seed(42)
        b = torch.randn(10, device=device)
        assert torch.allclose(a, b), "Seed setting failed!"
        print("Seed reproducibility test passed!")

        # Test NPU count
        npu_count = get_npu_count()
        print(f"\nAvailable NPUs: {npu_count}")

        print("\n" + "=" * 50)
        print("All device tests passed!")
        print("=" * 50)

    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nThis architecture requires Huawei Ascend NPUs.")
