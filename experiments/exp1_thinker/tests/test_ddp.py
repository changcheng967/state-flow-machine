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
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp


# HCCL requires these environment variables for correct multi-NPU operation
HCCL_ENV_VARS = {
    "TASK_QUEUE_ENABLE": "2",
    "CPU_AFFINITY_CONF": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_DETERMINISTIC": "true",
}


def setup_hccl_env(rank: int):
    """Set HCCL environment variables (ASCEND_RT_VISIBLE_DEVICES handled in main)."""
    for key, val in HCCL_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = val


def try_init_process_group(backend: str, rank: int, world_size: int, retries: int = 2, delay: int = 10):
    """Try initializing process group with retries for HCCL."""
    for attempt in range(retries):
        try:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                init_method="env://"
            )
            dist.barrier()
            return True
        except RuntimeError as e:
            if attempt < retries - 1 and "hccl" in str(e).lower():
                print(f"Rank {rank}: HCCL init failed (attempt {attempt+1}), "
                      f"retrying in {delay}s... Error: {e}", flush=True)
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
                time.sleep(delay)
            else:
                raise
    return False


def worker(rank: int, world_size: int, backend: str):
    """DDP worker function."""
    import torch_npu

    # Set device BEFORE process group init (critical for HCCL)
    setup_hccl_env(rank)
    # When ASCEND_RT_VISIBLE_DEVICES=rank, each rank sees exactly 1 NPU at index 0
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    try:
        # Initialize process group with retries
        try:
            try_init_process_group(backend, rank, world_size)
            print(f"Rank {rank}: {backend} backend initialized on {device}", flush=True)
        except RuntimeError as e:
            if backend == "hccl":
                print(f"Rank {rank}: HCCL failed after retries, trying gloo...", flush=True)
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
                backend = "gloo"
                try_init_process_group(backend, rank, world_size)
                print(f"Rank {rank}: gloo backend initialized (HCCL unavailable)", flush=True)
            else:
                raise

        # Create model on this rank's NPU
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)

        dist.barrier()  # all ranks have model ready before DDP wrap

        # Wrap with DDP (device_ids uses local index, always 0 with ASCEND_RT_VISIBLE_DEVICES)
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[0], output_device=0
        )

        # Create dummy batch
        batch_size = 16
        torch.manual_seed(42)  # same data on all ranks
        inputs = torch.randn(batch_size, 512, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)

        # Optimizer
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

        dist.barrier()  # all ranks ready before training

        # Training loop: 10 steps
        train_ok = True
        try:
            for step in range(10):
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                print(f"  Rank {rank} Step {step}: loss={loss.item():.4f}", flush=True)
        except Exception as train_err:
            print(f"Rank {rank} (physical npu:{rank}) -- TRAINING FAIL: {train_err}", flush=True)
            train_ok = False

        # All-reduce verification
        dist.barrier()
        test_tensor = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2  # sum(1..world_size)
        all_reduce_ok = abs(test_tensor.item() - expected) < 0.01

        dist.barrier()

        if all_reduce_ok and train_ok:
            print(f"Rank {rank} (physical npu:{rank}) -- DDP PASS "
                  f"(10 steps completed, all-reduce verified: "
                  f"got {test_tensor.item():.1f}, expected {expected:.1f}, "
                  f"backend={backend})",
                  flush=True)
        else:
            print(f"Rank {rank} (physical npu:{rank}) -- DDP PARTIAL "
                  f"(train={'ok' if train_ok else 'FAIL'}, "
                  f"all_reduce={'ok' if all_reduce_ok else 'FAIL'}, "
                  f"backend={backend})",
                  flush=True)

        dist.destroy_process_group()

    except Exception as e:
        print(f"Rank {rank} (physical npu:{rank}) -- DDP FAIL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_with_torchrun(backend: str):
    """Standard torchrun entry point -- env vars set by torchrun."""
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
    # CRITICAL: Set ASCEND_RT_VISIBLE_DEVICES BEFORE importing torch_npu.
    # torch_npu initializes the driver on first import, and the visible
    # devices cannot be changed after that.
    is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_torchrun and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(local_rank)
        # Also set HCCL env vars before import
        for key, val in HCCL_ENV_VARS.items():
            if key not in os.environ:
                os.environ[key] = val

    print("=" * 60)
    print("Thinker-3B Hardware Validation: Multi-NPU DDP")
    print("=" * 60)

    import torch_npu
    n_devices = torch.npu.device_count()
    print(f"NPU devices available (visible to this rank): {n_devices}")
    if is_torchrun:
        print(f"RANK={os.environ['RANK']}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
              f"WORLD_SIZE={os.environ['WORLD_SIZE']}")

    backend = "hccl"
    world_size = int(os.environ.get("WORLD_SIZE", n_devices))

    if is_torchrun:
        run_with_torchrun(backend)
    else:
        print(f"Manual spawn mode (world_size={world_size})")
        run_manual_spawn(backend, world_size)


if __name__ == "__main__":
    main()
