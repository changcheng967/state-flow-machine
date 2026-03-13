"""
Device selection for SFM - Ascend NPU only.

This architecture is designed for Huawei Ascend NPUs.
torch_npu is required.
"""

import torch
from typing import Tuple


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

        # Test tensor creation on device
        print("\nTesting tensor operations...")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        print(f"Matrix multiplication successful: result shape = {z.shape}")

        # Test seed setting
        set_seed(42)
        a = torch.randn(10, device=device)
        set_seed(42)
        b = torch.randn(10, device=device)
        assert torch.allclose(a, b), "Seed setting failed!"
        print("Seed reproducibility test passed!")

        print("\n" + "=" * 50)
        print("All device tests passed!")
        print("=" * 50)

    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nThis architecture requires Huawei Ascend NPUs.")
