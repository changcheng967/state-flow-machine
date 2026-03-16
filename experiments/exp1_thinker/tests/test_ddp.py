"""
test_ddp.py - Multi-NPU DDP validation for Thinker-3B

Tests DistributedDataParallel across all 4 NPUs with HCCL backend.
Falls back to gloo if HCCL is unavailable.

Usage:
    torchrun --nproc_per_node=4 experiments/exp1_thinker/tests/test_ddp.py

    # If torchrun fails, manual spawn:
    python experiments/exp1_thinker/tests/test_ddp.py --manual_spawn
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, backend: str):
    """DDP worker function."""
    try:
        # Initialize process group
        if backend == "hccl":
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                init_method="env://"
            )
        else:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                init_method="env://"
            )

        device = torch.device(f"npu:{rank}")

        # Create model on this rank's NPU
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)

        # Wrap with DDP
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

        # Create dummy batch
        batch_size = 16
        inputs = torch.randn(batch_size, 512, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)

        # Optimizer
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

        # Training loop: 10 steps
        for step in range(10):
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 0 and step == 0:
                print(f"  Step {step}: loss={loss.item():.4f}", flush=True)

        # All-reduce verification
        test_tensor = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2  # sum(1..world_size)
        all_reduce_ok = abs(test_tensor.item() - expected) < 0.01

        if all_reduce_ok:
            print(f"Rank {rank} on npu:{rank} -- DDP PASS "
                  f"(10 steps completed, all-reduce verified: "
                  f"got {test_tensor.item():.1f}, expected {expected:.1f})",
                  flush=True)
        else:
            print(f"Rank {rank} on npu:{rank} -- DDP PARTIAL "
                  f"(10 steps completed, all-reduce MISMATCH: "
                  f"got {test_tensor.item():.1f}, expected {expected:.1f})",
                  flush=True)

        dist.destroy_process_group()

    except Exception as e:
        print(f"Rank {rank} on npu:{rank} -- DDP FAIL: {e}", flush=True)
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_with_torchrun(backend: str):
    """Standard torchrun entry point — env vars set by torchrun."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    worker(rank, world_size, backend)


def run_manual_spawn(backend: str, world_size: int = 4):
    """Fallback: manual multiprocess spawn."""
    print(f"Manual spawn with {world_size} processes, backend={backend}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        fn=lambda rank, _: worker(rank, world_size, backend),
        args=(None,),
        nprocs=world_size,
        join=True
    )


def main():
    print("=" * 60)
    print("Thinker-3B Hardware Validation: Multi-NPU DDP")
    print("=" * 60)

    import torch_npu
    n_devices = torch.npu.device_count()
    print(f"NPU devices available: {n_devices}")

    # Determine if launched via torchrun or manual
    is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    backend = "hccl"
    world_size = int(os.environ.get("WORLD_SIZE", n_devices))

    if is_torchrun:
        print(f"Launched via torchrun: RANK={os.environ['RANK']}, "
              f"WORLD_SIZE={os.environ['WORLD_SIZE']}")
        print(f"Attempting HCCL backend...")
        try:
            run_with_torchrun(backend)
        except Exception as e:
            print(f"HCCL failed: {e}")
            print(f"Falling back to gloo backend...")
            run_with_torchrun("gloo")
    else:
        print(f"Manual spawn mode (world_size={world_size})")
        print(f"Attempting HCCL backend...")
        try:
            run_manual_spawn(backend, world_size)
        except Exception as e:
            print(f"HCCL failed: {e}")
            print(f"Falling back to gloo backend...")
            run_manual_spawn("gloo", world_size)


if __name__ == "__main__":
    main()
