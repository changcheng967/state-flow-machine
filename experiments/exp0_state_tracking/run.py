"""
Run Script for State Tracking Experiment (Exp 0)

End-to-end script that:
1. Generates data
2. Trains Execution System (State Slots)
3. Trains Transformer baseline
4. Evaluates generalization to longer sequences
5. Reports comparison

PASS = State Slots generalize to 4x program length, transformer doesn't.
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig, ExperimentConfig
from sfm.utils.device import get_device, set_seed
from train import run_experiment
from evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Run State Tracking Experiment (Exp 0)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with minimal samples"
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only train models, skip evaluation"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate existing models"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/exp0",
        help="Directory to save results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--base_length",
        type=int,
        default=10,
        help="Base program length for generalization test"
    )

    args = parser.parse_args()

    # Setup
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print("STATE-FLOW MACHINE EXPERIMENT 0: STATE TRACKING")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Save directory: {args.save_dir}")

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

        training_results = run_experiment(
            config=exp_config,
            sfm_config=sfm_config,
            device=device,
            save_dir=args.save_dir
        )

    # Evaluation phase
    if not args.train_only:
        print("\n" + "=" * 70)
        print("PHASE 2: EVALUATION")
        print("=" * 70)

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

    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {args.save_dir}")

    # Print final comparison if we have evaluation results
    if not args.train_only and 'eval_results' in dir():
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
                status = "✓ PASS"
            elif mult == 4 and delta <= 0:
                status = "✗ FAIL"

            print(f"{mult}x{'':<10} {exec_acc:<15.4f} {trans_acc:<15.4f} {delta:+.4f}    {status}")

        # Final verdict
        print("\n" + "-" * 70)
        if 4 in multipliers:
            exec_at_4x = exec_results.get(4, {}).get("accuracy", 0)
            trans_at_4x = trans_results.get(4, {}).get("accuracy", 0)

            if exec_at_4x > trans_at_4x + 0.05:
                print("VERDICT: ✓ PASS - State Slots significantly outperform transformer at 4x length")
            elif exec_at_4x > trans_at_4x:
                print("VERDICT: ~ MARGINAL - State Slots slightly better, needs more training")
            else:
                print("VERDICT: ✗ FAIL - Transformer matches or exceeds State Slots")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
