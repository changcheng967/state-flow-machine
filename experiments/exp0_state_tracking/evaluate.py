"""
Evaluation Script for State Tracking Experiment

Evaluates trained models on longer sequences to test generalization.
The key hypothesis: State Slots generalize to 4x length, transformers don't.
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
from train import StateTrackingWrapper
from baseline_transformer import TransformerEncoderOnly
from generate_data import SimpleProgramGenerator
from sfm.tokenizer.code_tokenizer import CodeTokenizer


class Evaluator:
    """
    Evaluates models on generalization to longer sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CodeTokenizer,
        device: torch.device,
        model_name: str = "model"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name

    @torch.no_grad()
    def evaluate_on_length(
        self,
        programs: List[Dict],
        max_length: int = 256
    ) -> Dict[str, float]:
        """
        Evaluate model on programs of a specific length.

        Args:
            programs: List of program dicts.
            max_length: Maximum tokenized length.

        Returns:
            Metrics dict.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        for sample in programs:
            program = sample["program"]
            final_value = sample["final_value"]

            # Tokenize
            text = "\n".join(program)
            ids = self.tokenizer.encode(text)[:max_length]

            # Pad
            pad_id = self.tokenizer.special_tokens["<pad>"]
            while len(ids) < max_length:
                ids.append(pad_id)

            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                [[1 if id != pad_id else 0 for id in ids]],
                dtype=torch.long,
                device=self.device
            )

            # Forward pass
            logits = self.model(input_ids, attention_mask)

            # Compute metrics
            pred = logits.argmax(dim=-1).item()

            # For loss, we need to handle potentially large value range
            # Use regression-style loss for values > num_classes
            if final_value < logits.size(-1):
                labels = torch.tensor([final_value], dtype=torch.long, device=self.device)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                if pred == final_value:
                    correct += 1
            else:
                # For out-of-range values, check relative error
                error = abs(pred - final_value)
                if error < final_value * 0.1:  # Within 10%
                    correct += 0.5  # Partial credit

            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "loss": total_loss / total if total > 0 else 0,
            "total": total
        }

    def evaluate_generalization(
        self,
        base_length: int,
        multipliers: List[int] = [1, 2, 4, 8],
        samples_per_length: int = 100,
        seed: int = 42
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate generalization to longer sequences.

        Args:
            base_length: Base program length.
            multipliers: Length multipliers to test.
            samples_per_length: Samples per length.
            seed: Random seed.

        Returns:
            Dict mapping length multiplier to metrics.
        """
        results = {}
        generator = SimpleProgramGenerator(
            num_variables=5,
            max_program_length=base_length * 8,  # Max multiplier
            seed=seed
        )

        print(f"\nEvaluating {self.model_name} on generalization...")

        for mult in multipliers:
            target_length = base_length * mult
            print(f"  Length {mult}x ({target_length} statements)...")

            # Generate programs at this length
            programs = []
            for _ in range(samples_per_length):
                program, target, final = generator.generate_program()
                # Filter by length
                if len(program) >= target_length - 2 and len(program) <= target_length + 2:
                    programs.append({
                        "program": program,
                        "target_variable": target,
                        "final_value": final
                    })

            # Evaluate
            metrics = self.evaluate_on_length(programs)
            results[mult] = metrics

            print(f"    Accuracy: {metrics['accuracy']:.4f}")

        return results


def load_model(
    model_path: str,
    model_type: str,
    config: SFMConfig,
    vocab_size: int,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """
    Load a trained model.

    Args:
        model_path: Path to checkpoint.
        model_type: "execution" or "transformer".
        config: SFM config.
        vocab_size: Vocabulary size.
        num_classes: Number of output classes.
        device: Device.

    Returns:
        Loaded model.
    """
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
        model = StateTrackingWrapper(execution, vocab_size, num_classes)
    else:
        model = TransformerEncoderOnly(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=4,
            num_layers=4,
            d_ff=config.d_model * 4,
            num_output_classes=num_classes
        )

    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model path {model_path} not found, using random weights")

    model = model.to(device)
    return model


def run_evaluation(
    save_dir: str,
    base_length: int = 20,
    multipliers: List[int] = [1, 2, 4],
    samples_per_length: int = 50,
    quick: bool = False
) -> Dict:
    """
    Run full evaluation.

    Args:
        save_dir: Directory with saved models.
        base_length: Base program length.
        multipliers: Length multipliers.
        samples_per_length: Samples per length.
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

    # Load tokenizer (recreate from data)
    data_dir = os.path.join(os.path.dirname(save_dir), "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(save_dir, "data")

    # If no data, create tokenizer from scratch
    if os.path.exists(os.path.join(data_dir, "train.json")):
        with open(os.path.join(data_dir, "train.json"), 'r') as f:
            train_data = json.load(f)
        corpus = ["\n".join(s["program"]) for s in train_data]
    else:
        # Generate sample programs for tokenizer
        gen = SimpleProgramGenerator(seed=42)
        programs = gen.generate_dataset(1000)
        corpus = ["\n".join(p["program"]) for p in programs]

    tokenizer = CodeTokenizer(vocab_size=5000, min_freq=1)
    tokenizer.train(corpus, verbose=False)

    results = {}

    # Evaluate Execution System
    exec_path = os.path.join(save_dir, "execution_best.pt")
    if os.path.exists(exec_path) or True:  # Always try, will use random weights if not found
        print("\n" + "=" * 60)
        print("Evaluating Execution System (State Slots)")
        print("=" * 60)

        exec_model = load_model(
            exec_path,
            "execution",
            config,
            tokenizer.vocab_size_actual,
            500,
            device
        )

        exec_evaluator = Evaluator(exec_model, tokenizer, device, "Execution System")
        exec_results = exec_evaluator.evaluate_generalization(
            base_length=base_length,
            multipliers=multipliers,
            samples_per_length=samples_per_length
        )
        results["execution"] = exec_results

    # Evaluate Transformer
    trans_path = os.path.join(save_dir, "transformer_best.pt")
    if os.path.exists(trans_path) or True:
        print("\n" + "=" * 60)
        print("Evaluating Transformer Baseline")
        print("=" * 60)

        trans_model = load_model(
            trans_path,
            "transformer",
            config,
            tokenizer.vocab_size_actual,
            500,
            device
        )

        trans_evaluator = Evaluator(trans_model, tokenizer, device, "Transformer")
        trans_results = trans_evaluator.evaluate_generalization(
            base_length=base_length,
            multipliers=multipliers,
            samples_per_length=samples_per_length
        )
        results["transformer"] = trans_results

    # Print comparison
    print("\n" + "=" * 60)
    print("GENERALIZATION COMPARISON")
    print("=" * 60)

    print(f"\n{'Length':<10} {'Execution':<15} {'Transformer':<15} {'Gap':<10}")
    print("-" * 50)

    for mult in multipliers:
        exec_acc = results["execution"][mult]["accuracy"]
        trans_acc = results["transformer"][mult]["accuracy"]
        gap = exec_acc - trans_acc

        print(f"{mult}x{'':<8} {exec_acc:<15.4f} {trans_acc:<15.4f} {gap:+.4f}")

    # Save results
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate state tracking models")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0",
                        help="Directory with saved models")
    parser.add_argument("--base_length", type=int, default=20,
                        help="Base program length")
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4],
                        help="Length multipliers to test")
    parser.add_argument("--samples", type=int, default=50,
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
