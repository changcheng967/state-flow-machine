"""
Evaluation Script for State Tracking Experiment - REGRESSION VERSION

Evaluates trained models on longer sequences to test generalization.
The key hypothesis: State Slots generalize to 4x length, transformers don't.

FIXED FOR REGRESSION:
- Models output single sigmoid scalar [0,1], not 500-class softmax
- Exact match: (prediction * 100).round() == final_value
- Close match: abs(prediction * 100 - final_value) < 5
- MSE: computed on normalized values
- Uses SimpleTokenizer (matching train.py)
- Generates test programs at exact target lengths
- Saves matplotlib plot to outputs/exp0/length_generalization.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig
from sfm.systems.execution import ExecutionSystem
from sfm.utils.device import get_device, set_seed
from sfm.tokenizer.code_tokenizer import SimpleTokenizer

# Import NPU-compatible model classes
from train import StateTrackingWrapper
from baseline_transformer import TransformerEncoderOnly


class Evaluator:
    """Evaluator for REGRESSION models with batched NPU-efficient inference."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: SimpleTokenizer,
        device: torch.device,
        model_name: str = "model",
        is_transformer: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.is_transformer = is_transformer

    @torch.no_grad()
    def evaluate_samples(
        self,
        samples: List[Dict],
        max_length: int = 512,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on samples with BATCHED inference for NPU efficiency.

        Reduces N forward passes to N/32, which is 30x faster on NPU.

        Returns:
            Metrics: exact_match, close_match, mse, mae
        """
        self.model.eval()
        num_samples = len(samples)
        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)

        # Step 1: Tokenize all samples and pad to max_length
        all_ids = []
        all_masks = []
        final_values = []

        for sample in samples:
            program = sample["program"]
            final_value = sample["final_value"]  # Raw value [0, 100]

            # Tokenize
            text = "\n".join(program)
            ids = self.tokenizer.encode(text)[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(ids)

            # Pad to max_length
            while len(ids) < max_length:
                ids.append(pad_id)
                mask.append(0)

            all_ids.append(ids)
            all_masks.append(mask)
            final_values.append(final_value)

        # Stack into tensors on CPU first (avoid NPU memory pressure)
        all_ids = torch.tensor(all_ids, dtype=torch.long)
        all_masks = torch.tensor(all_masks, dtype=torch.long)
        final_values_tensor = torch.tensor(final_values, dtype=torch.float32)

        # Step 2: Process in batches
        all_predictions = []

        for i in range(0, num_samples, batch_size):
            batch_ids = all_ids[i:i+batch_size].to(self.device)
            batch_mask = all_masks[i:i+batch_size].to(self.device)

            # Forward pass
            if self.is_transformer:
                mask = (batch_mask == 0)  # True for padding
                predictions = self.model(batch_ids, mask=mask)
            else:
                predictions = self.model(batch_ids, attention_mask=batch_mask)

            all_predictions.append(predictions.cpu())

        # Concatenate all predictions
        all_preds = torch.cat(all_predictions, dim=0)  # (N,)

        # Step 3: Compute metrics on full array
        predicted_values = all_preds * 100  # Convert from [0,1] to [0,100]
        final_values_norm = final_values_tensor / 100.0

        # Exact match: round(predicted) == final_value
        exact_match = (predicted_values.round() == final_values_tensor).float().mean().item()

        # Close match: abs(predicted - final) < 5
        close_match = ((predicted_values - final_values_tensor).abs() < 5).float().mean().item()

        # MSE on normalized values
        mse = ((all_preds - final_values_norm).pow(2)).mean().item()

        # MAE on raw values
        mae = (predicted_values - final_values_tensor).abs().mean().item()

        return {
            "exact_match": exact_match,
            "close_match": close_match,
            "mse": mse,
            "mae": mae,
            "total": num_samples
        }


def generate_test_programs(
    num_samples: int,
    exact_length: int,
    seed: int = 42,
    difficulty: str = "easy"
) -> List[Dict]:
    """
    Generate test programs at an EXACT length.

    Args:
        num_samples: Number of programs to generate.
        exact_length: Exact number of operations (not including init and query lines).
        seed: Random seed.
        difficulty: Difficulty level.

    Returns:
        List of samples with programs at the exact target length.
    """
    from generate_data import SimpleProgramGenerator

    # Create generator with reasonable defaults - exact_length passed to generate_program()
    generator = SimpleProgramGenerator(
        num_variables=5,
        max_program_length=exact_length,  # Upper bound
        min_program_length=1,  # Lower bound (will be overridden by exact_length param)
        seed=seed,
        difficulty=difficulty
    )

    samples = []
    for _ in range(num_samples):
        # Use exact_length parameter directly - makes intent explicit
        program, target, final_value = generator.generate_program(exact_length=exact_length)
        samples.append({
            "program": program,
            "target_variable": target,
            "final_value": final_value
        })

    return samples


def load_model(
    model_path: str,
    model_type: str,
    config: SFMConfig,
    vocab_size: int,
    device: torch.device
) -> nn.Module:
    """Load a trained model for REGRESSION."""
    if model_type == "execution":
        execution = ExecutionSystem(
            input_dim=config.d_model,
            hidden_dim=config.deltanet_hidden_dim,
            num_slots=config.execution_num_slots,
            slot_dim=config.execution_slot_dim,
            max_ticks=config.execution_max_ticks,
            num_heads=config.execution_num_heads,
            dropout=config.dropout
        )
        model = StateTrackingWrapper(execution, vocab_size)
    else:
        model = TransformerEncoderOnly(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=4,
            num_layers=4,
            d_ff=config.d_model * 4
        )

    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model path {model_path} not found, using random weights")

    model = model.to(device)
    model.eval()
    return model


def run_evaluation(
    save_dir: str,
    base_length: int = 10,
    multipliers: List[int] = [1, 2, 4, 8],
    samples_per_length: int = 100,
    quick: bool = False
) -> Dict:
    """
    Run full evaluation with exact-length test programs.

    Args:
        save_dir: Directory with saved models.
        base_length: Base program length (number of operations).
        multipliers: Length multipliers to test.
        samples_per_length: Samples per length multiplier.
        quick: Quick test mode.

    Returns:
        Evaluation results.
    """
    set_seed(42)
    device = get_device()

    if quick:
        samples_per_length = 10
        multipliers = [1, 2]

    # Load config
    config = SFMConfig.small()

    # Load tokenizer from saved vocabulary - HARD FAIL if missing
    vocab_path = os.path.join(save_dir, "tokenizer_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Tokenizer vocabulary not found at {vocab_path}. "
            "Training must be run first to generate tokenizer_vocab.json. "
            "Falling back to recreation is disabled for reproducibility."
        )
    print(f"Loading tokenizer from {vocab_path}")
    tokenizer = SimpleTokenizer.load(vocab_path)

    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size_actual}")

    results = {"execution": {}, "transformer": {}}

    # Evaluate both models
    for model_type, model_name in [("execution", "State Slots"), ("transformer", "Transformer")]:
        model_path = os.path.join(save_dir, f"{model_type}_best.pt")
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")

        model = load_model(model_path, model_type, config, tokenizer.vocab_size_actual, device)
        evaluator = Evaluator(model, tokenizer, device, model_name, is_transformer=(model_type == "transformer"))

        for mult in multipliers:
            target_length = base_length * mult
            print(f"\n  Length {mult}x ({target_length} operations)...")

            # Generate test programs at EXACT target length
            samples = generate_test_programs(
                num_samples=samples_per_length,
                exact_length=target_length,
                seed=42 + mult,  # Different seed per length
                difficulty="easy"
            )

            # Evaluate
            metrics = evaluator.evaluate_samples(samples)
            results[model_type][mult] = metrics

            print(f"    Exact Match: {metrics['exact_match']:.4f}")
            print(f"    Close Match: {metrics['close_match']:.4f}")
            print(f"    MSE: {metrics['mse']:.6f}")
            print(f"    MAE: {metrics['mae']:.2f}")

    # Print results table
    print("\n" + "=" * 80)
    print("GENERALIZATION RESULTS")
    print("=" * 80)
    print(f"\n{'Length':<8} {'SS EMA':<10} {'TF EMA':<10} {'SS Close':<10} {'TF Close':<10} {'SS MSE':<10} {'TF MSE':<10}")
    print("-" * 78)

    for mult in multipliers:
        ss = results["execution"][mult]
        tf = results["transformer"][mult]
        print(f"{mult}x{'':<6} {ss['exact_match']:<10.4f} {tf['exact_match']:<10.4f} "
              f"{ss['close_match']:<10.4f} {tf['close_match']:<10.4f} "
              f"{ss['mse']:<10.6f} {tf['mse']:<10.6f}")

    # Compute generalization ratios
    print("\n" + "-" * 50)
    print("GENERALIZATION RATIOS (EMA_4x / EMA_1x):")
    if 1 in multipliers and 4 in multipliers:
        ss_ratio = results["execution"][4]["exact_match"] / max(results["execution"][1]["exact_match"], 0.001)
        tf_ratio = results["transformer"][4]["exact_match"] / max(results["transformer"][1]["exact_match"], 0.001)
        print(f"  State Slots: {ss_ratio:.2f}x")
        print(f"  Transformer: {tf_ratio:.2f}x")
        if ss_ratio > tf_ratio + 0.1:
            print(f"  VERDICT: PASS - State Slots generalize better")
        elif ss_ratio > tf_ratio:
            print(f"  VERDICT: MARGINAL - Slight advantage")
        else:
            print(f"  VERDICT: FAIL - No advantage")

    # Save results
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate matplotlib plot
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ss_accs = [results["execution"][m]["exact_match"] * 100 for m in multipliers]
        tf_accs = [results["transformer"][m]["exact_match"] * 100 for m in multipliers]
        x_labels = [f"{m}x" for m in multipliers]

        ax.plot(x_labels, ss_accs, 'b-o', linewidth=2, markersize=8, label='State Slots')
        ax.plot(x_labels, tf_accs, 'r--s', linewidth=2, markersize=8, label='Transformer')

        ax.set_xlabel('Length Multiplier', fontsize=12)
        ax.set_ylabel('Exact Match Accuracy (%)', fontsize=12)
        ax.set_title('Experiment 0: Length Generalization', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "length_generalization.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate state tracking models (REGRESSION)")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0",
                        help="Directory with saved models")
    parser.add_argument("--base_length", type=int, default=10,
                        help="Base program length (number of operations)")
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Length multipliers to test")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per length")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode")
    args = parser.parse_args()

    results = run_evaluation(
        save_dir=args.save_dir,
        base_length=args.base_length,
        multipliers=args.multipliers,
        samples_per_length=args.samples,
        quick=args.quick
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
