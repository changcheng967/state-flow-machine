"""
Run Script for State Tracking Experiment (Exp 0)

End-to-end script that:
1. Generates data
2. Trains Execution System (State Slots)
3. Trains Transformer baseline
4. Evaluates generalization to longer sequences
5. Reports comparison with timing breakdown

PASS = State Slots generalize to 4x program length, transformer doesn't.

USAGE:
    python run.py --quick                              # smoke test, 1 NPU
    python run.py --npus 4 --epochs 50                 # 4 NPUs
    python run.py --npus 4 --epochs 50 --samples 50000 # full run
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig, ExperimentConfig
from sfm.utils.device import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Run State Tracking Experiment (Exp 0)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with minimal samples")
    parser.add_argument("--train_only", action="store_true",
                        help="Only train models, skip evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing models")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0",
                        help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--base_length", type=int, default=10,
                        help="Base program length for generalization test")
    parser.add_argument("--npus", type=int, default=1,
                        help="Number of NPUs for distributed training")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Data difficulty level")

    args = parser.parse_args()

    # Setup
    set_seed(42)

    print("=" * 70)
    print("STATE-FLOW MACHINE EXPERIMENT 0: STATE TRACKING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  NPUs: {args.npus}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training samples: {args.samples}")
    print(f"  Difficulty: {args.difficulty}")

    if args.quick:
        print("\n*** QUICK MODE - Minimal samples for smoke test ***")
        exp_config = ExperimentConfig.quick()
        sfm_config = SFMConfig.small()
        exp_config.num_epochs = 1
        base_length = 5
        multipliers = [1, 2]
    else:
        exp_config = ExperimentConfig(
            name="exp0_state_tracking",
            train_samples=args.samples,
            val_samples=args.samples // 10,
            max_program_length=20,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        sfm_config = SFMConfig.small()
        base_length = args.base_length
        multipliers = [1, 2, 4]

    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()

    # Training phase
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING")
        print("=" * 70)

        if args.npus > 1:
            # Multi-NPU training
            exit_code = run_distributed_training(exp_config, sfm_config, args)
            if exit_code != 0:
                print(f"\n[ERROR] Distributed training failed with exit code {exit_code}")
                print("Check the error messages above. Exiting...")
                return exit_code
        else:
            # Single NPU training
            device = get_device()
            from train import run_experiment
            training_results = run_experiment(
                config=exp_config,
                sfm_config=sfm_config,
                device=device,
                save_dir=args.save_dir,
                distributed=False,
                rank=0,
                world_size=1
            )

    # Synchronize NPU state before evaluation
    if not args.train_only:
        print("\nSynchronizing NPU state before evaluation...")
        try:
            import torch
            torch.npu.synchronize()
            time.sleep(2)  # Brief pause for NPU state cleanup
        except Exception as e:
            print(f"Note: NPU sync skipped ({e})")

    # Evaluation phase
    eval_results = None
    if not args.train_only:
        print("\n" + "=" * 70)
        print("PHASE 2: EVALUATION")
        print("=" * 70)

        from evaluate import run_evaluation
        eval_results = run_evaluation(
            save_dir=args.save_dir,
            base_length=base_length,
            multipliers=multipliers,
            samples_per_length=50 if not args.quick else 10,
            quick=args.quick
        )

    # Summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"\nTotal time: {total_time / 60:.1f} minutes ({total_time:.1f}s)")
    print(f"Results saved to: {args.save_dir}")

    # Print timing breakdown if available
    results_path = os.path.join(args.save_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        if "timings" in results:
            print("\n--- Timing Breakdown ---")
            timings = results["timings"]
            for name, t in timings.items():
                print(f"  {name}: {t:.2f}s")

    # Print final comparison if we have evaluation results
    if eval_results and not args.train_only:
        print("\n" + "-" * 70)
        print("GENERALIZATION RESULTS")
        print("-" * 70)

        exec_results = eval_results.get("execution", {})
        trans_results = eval_results.get("transformer", {})

        print(f"\n{'Length':<12} {'State Slots':<15} {'Transformer':<15} {'Delta':<10}")
        print("-" * 52)

        for mult in multipliers:
            exec_acc = exec_results.get(mult, {}).get("accuracy", 0)
            trans_acc = trans_results.get(mult, {}).get("accuracy", 0)
            delta = exec_acc - trans_acc

            status = ""
            if mult == 4 and delta > 0.05:
                status = "[OK] PASS"
            elif mult == 4 and delta <= 0:
                status = "[X] FAIL"

            print(f"{mult}x{'':<10} {exec_acc:<15.4f} {trans_acc:<15.4f} {delta:+.4f}    {status}")

        # Final verdict
        print("\n" + "-" * 70)
        if 4 in multipliers:
            exec_at_4x = exec_results.get(4, {}).get("accuracy", 0)
            trans_at_4x = trans_results.get(4, {}).get("accuracy", 0)

            if exec_at_4x > trans_at_4x + 0.05:
                print("VERDICT: [OK] PASS - State Slots significantly outperform transformer at 4x length")
            elif exec_at_4x > trans_at_4x:
                print("VERDICT: [~] MARGINAL - State Slots slightly better, needs more training")
            else:
                print("VERDICT: [X] FAIL - Transformer matches or exceeds State Slots")

    print("\n" + "=" * 70)

    return 0


def run_distributed_training(exp_config, sfm_config, args):
    """Run training with multiple NPUs using torchrun."""
    import subprocess

    print(f"\nLaunching distributed training on {args.npus} NPUs...")

    # Use torchrun (NOT deprecated torch.distributed.launch)
    # torchrun sets LOCAL_RANK via environment variable instead of CLI arg
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.npus}",
        "--master_port=29500",
        os.path.join(os.path.dirname(__file__), "train.py"),
        f"--save_dir={args.save_dir}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--samples={args.samples}",
    ]

    if args.quick:
        cmd.append("--quick")

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
