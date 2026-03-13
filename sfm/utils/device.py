"""
Automatic device selection for SFM.

Tries torch_npu first (Huawei Ascend), falls back to CUDA, then CPU.
No user configuration needed.
"""

import torch
from typing import Tuple, Optional


def get_device() -> torch.device:
    """
    Get the best available device automatically.

    Priority: torch_npu (Ascend) > CUDA > CPU

    Returns:
        torch.device: The best available device.
    """
    # Try torch_npu first (Huawei Ascend NPUs)
    try:
        import torch_npu
        if torch.npu.is_available():
            device = torch.device("npu:0")
            print(f"[Device] Using Ascend NPU: {torch.npu.get_device_name(0)}")
            return device
    except ImportError:
        pass
    except Exception as e:
        print(f"[Device] torch_npu import failed: {e}")

    # Try CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    # Fall back to CPU
    print("[Device] Using CPU (no accelerator available)")
    return torch.device("cpu")


def get_device_info() -> Tuple[str, int, Optional[str]]:
    """
    Get detailed device information.

    Returns:
        Tuple of (device_type, memory_gb, device_name)
    """
    # Try torch_npu
    try:
        import torch_npu
        if torch.npu.is_available():
            memory_gb = torch.npu.get_device_properties(0).total_memory / (1024**3)
            name = torch.npu.get_device_name(0)
            return ("npu", int(memory_gb), name)
    except ImportError:
        pass
    except Exception:
        pass

    # Try CUDA
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        name = torch.cuda.get_device_name(0)
        return ("cuda", int(memory_gb), name)

    # CPU
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    return ("cpu", int(memory_gb), "CPU")


def set_seed(seed: int, device: Optional[torch.device] = None) -> None:
    """
    Set random seeds for reproducibility across all devices.

    Args:
        seed: Random seed value.
        device: Optional device to set seed for (for CUDA/NPU).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is not None:
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif device.type == "npu":
            try:
                import torch_npu
                torch.npu.manual_seed_all(seed)
            except ImportError:
                pass


def to_device(data, device: torch.device):
    """
    Recursively move data to device.

    Handles tensors, dicts, lists, and tuples.

    Args:
        data: Data to move (tensor, dict, list, or tuple).
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
    print("SFM Device Selection Test")
    print("=" * 50)

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
    set_seed(42, device)
    a = torch.randn(10, device=device)
    set_seed(42, device)
    b = torch.randn(10, device=device)
    assert torch.allclose(a, b), "Seed setting failed!"
    print("Seed reproducibility test passed!")

    print("\n" + "=" * 50)
    print("All device tests passed!")
    print("=" * 50)
