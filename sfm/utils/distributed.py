"""
Distributed Training Utilities for SFM on Ascend NPUs

Supports:
- Single NPU training
- DataParallel (DP) - single process, multiple NPUs
- DistributedDataParallel (DDP) - multi-process, multiple NPUs
- Automatic world size detection
- Gradient synchronization
- Mixed precision training (AMP)
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Any, Dict
import math


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of NPUs in distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank on current node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "hccl") -> Tuple[int, int, torch.device]:
    """
    Setup distributed training environment.

    Args:
        backend: Distributed backend ('hccl' for Ascend NPUs).

    Returns:
        Tuple of (rank, world_size, device).
    """
    # Get environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Initialize distributed process group
        dist.init_process_group(
            backend=backend,  # HCCL for Ascend NPUs
            init_method="env://"
        )

    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)

    if is_main_process():
        print(f"[Distributed] Rank {rank}/{world_size}, Local rank {local_rank}")
        print(f"[Distributed] Backend: {backend}")

    return rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_model(
    model: nn.Module,
    device: torch.device,
    distributed: bool = False,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True
) -> nn.Module:
    """
    Wrap model for parallel training.

    Args:
        model: Model to wrap.
        device: Target device.
        distributed: Whether to use DDP.
        find_unused_parameters: For models with unused parameters.
        broadcast_buffers: Whether to broadcast buffers.

    Returns:
        Wrapped model.
    """
    model = model.to(device)

    if distributed and is_distributed():
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers
        )
        if is_main_process():
            print("[Distributed] Model wrapped with DistributedDataParallel")

    return model


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a distributed-aware dataloader.

    Args:
        dataset: Dataset to load.
        batch_size: Per-GPU batch size.
        shuffle: Whether to shuffle (handled by sampler in distributed).
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        DataLoader with distributed sampler if needed.
    """
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle
        )
        # Don't shuffle in loader when using sampler
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


def all_reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce.
        average: Whether to average or sum.

    Returns:
        Reduced tensor.
    """
    if not is_distributed():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if average:
        tensor.div_(get_world_size())

    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.

    Args:
        tensor: Local tensor to gather.

    Returns:
        Gathered tensor from all processes.
    """
    if not is_distributed():
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)

    return torch.cat(gathered, dim=0)


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)
        for batch in dataloader:
            loss = model(batch) / accumulation_steps
            loss.backward()
            if accumulator.step():
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def step(self) -> bool:
        """
        Check if we should perform an optimizer step.

        Returns:
            True if should step, False otherwise.
        """
        self.step_count += 1
        if self.step_count >= self.accumulation_steps:
            self.step_count = 0
            return True
        return False

    def should_log(self) -> bool:
        """Check if we should log (after optimizer step)."""
        return self.step_count == 0


class MixedPrecisionTrainer:
    """
    Mixed precision training manager for Ascend NPUs.

    Handles automatic loss scaling and gradient unscaling.
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        self.enabled = enabled
        self.scaler = torch.npu.amp.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        ) if enabled else None

    def forward(
        self,
        model: nn.Module,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with automatic mixed precision.

        Args:
            model: Model to run.
            *args: Model arguments.
            **kwargs: Model keyword arguments.

        Returns:
            Model output.
        """
        if self.enabled:
            with torch.npu.amp.autocast():
                return model(*args, **kwargs)
        return model(*args, **kwargs)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = None,
        model: Optional[nn.Module] = None
    ):
        """
        Backward pass with gradient scaling.

        Args:
            loss: Loss tensor.
            optimizer: Optimizer.
            clip_grad_norm: Optional gradient clipping value.
            model: Model for gradient clipping (if DDP, use model.module).
        """
        if self.enabled:
            self.scaler.scale(loss).backward()

            if clip_grad_norm is not None and model is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_norm
                )

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()

            if clip_grad_norm is not None and model is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_norm
                )

            optimizer.step()


def get_effective_batch_size(
    per_device_batch_size: int,
    world_size: int = None,
    gradient_accumulation_steps: int = 1
) -> int:
    """
    Calculate effective batch size.

    Args:
        per_device_batch_size: Batch size per device.
        world_size: Number of devices (auto-detected if None).
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Effective batch size.
    """
    if world_size is None:
        world_size = get_world_size()

    return per_device_batch_size * world_size * gradient_accumulation_steps


def print_distributed_info():
    """Print distributed training information."""
    print("=" * 50)
    print("Distributed Training Information")
    print("=" * 50)
    print(f"  Distributed: {is_distributed()}")
    print(f"  World Size:  {get_world_size()}")
    print(f"  Rank:        {get_rank()}")
    print(f"  Local Rank:  {get_local_rank()}")
    print(f"  Main Process: {is_main_process()}")
    print(f"  NPUs Available: {torch.npu.device_count()}")
    for i in range(torch.npu.device_count()):
        print(f"    NPU {i}: {torch.npu.get_device_name(i)}")
    print("=" * 50)


if __name__ == "__main__":
    import torch_npu

    print_distributed_info()

    # Test gradient accumulator
    print("\nTesting GradientAccumulator...")
    acc = GradientAccumulator(accumulation_steps=4)
    steps = [acc.step() for _ in range(8)]
    print(f"  Steps should_trigger: {steps}")
    assert steps == [False, False, False, True, False, False, False, True]
    print("  [OK] GradientAccumulator test passed!")
