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


class StateTrackingWrapper(nn.Module):
    """Wraps ExecutionSystem for state tracking REGRESSION."""

    def __init__(
        self,
        execution_system: ExecutionSystem,
        vocab_size: int
    ):
        super().__init__()
        self.execution = execution_system
        self.embedding = nn.Embedding(vocab_size, execution_system.input_dim)
        # REGRESSION: Output single scalar in [0, 1]
        self.regressor = nn.Sequential(
            nn.LayerNorm(execution_system.input_dim),
            nn.Linear(execution_system.input_dim, execution_system.input_dim),
            nn.ReLU(),
            nn.Linear(execution_system.input_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.execution(x)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)  # (batch,)


class TransformerEncoderOnly(nn.Module):
    """Simple encoder-only transformer for baseline comparison - REGRESSION."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # REGRESSION: Output single scalar with sigmoid
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:, :x.size(1), :]
        if mask is not None:
            # mask is True for padding
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)
        if mask is not None:
            mask_weights = (~mask).unsqueeze(-1).float()
            pooled = (x * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class Evaluator:
    """Evaluator for REGRESSION models."""

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
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Evaluate model on samples.

        Returns:
            Metrics: exact_match, close_match, mse, mae
        """
        self.model.eval()
        total_exact = 0
        total_close = 0
        total_mse = 0
        total_mae = 0
        total = 0

        for sample in samples:
            program = sample["program"]
            final_value = sample["final_value"]  # Raw value [0, 100]

            # Tokenize
            text = "\n".join(program)
            ids = self.tokenizer.encode(text)[:max_length]

            # Pad
            pad_id = self.tokenizer.token_to_id.get("<pad>", 0)
            while len(ids) < max_length:
                ids.append(pad_id)

            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                [[1 if id != pad_id else 0 for id in ids]],
                dtype=torch.long,
                device=self.device
            )

            # Forward pass
            if self.is_transformer:
                mask = (attention_mask == 0)  # True for padding
                prediction = self.model(input_ids, mask=mask)
            else:
                prediction = self.model(input_ids, attention_mask=attention_mask)

            # Prediction is in [0, 1], convert to [0, 100]
            predicted_value = (prediction.item() * 100)
            final_value_norm = final_value / 100.0

            # Metrics - exact match requires rounding to correct integer
            exact_match = 1 if round(predicted_value) == final_value else 0
            close_match = 1 if abs(predicted_value - final_value) < 5 else 0

            total_exact += exact_match
            total_close += close_match
            total_mse += (prediction.item() - final_value_norm) ** 2
            total_mae += abs(predicted_value - final_value)
            total += 1

        return {
            "exact_match": total_exact / total if total > 0 else 0,
            "close_match": total_close / total if total > 0 else 0,
            "mse": total_mse / total if total > 0 else 0,
            "mae": total_mae / total if total > 0 else 0,
            "total": total
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

    generator = SimpleProgramGenerator(
        num_variables=5,
        max_program_length=exact_length,
        min_program_length=exact_length,  # EXACT length!
        seed=seed,
        difficulty=difficulty
    )

    samples = []
    for _ in range(num_samples):
        program, target, final_value = generator.generate_program()
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

    # Load tokenizer from saved vocabulary
    vocab_path = os.path.join(save_dir, "tokenizer_vocab.json")
    if os.path.exists(vocab_path):
        print(f"Loading tokenizer from {vocab_path}")
        tokenizer = SimpleTokenizer.load(vocab_path)
    else:
        # Fallback: recreate tokenizer from training data
        print("Tokenizer vocab not found, recreating from training data...")
        data_dir = os.path.join(save_dir, "data")
        if os.path.exists(os.path.join(data_dir, "train.json")):
            with open(os.path.join(data_dir, "train.json"), 'r') as f:
                train_data = json.load(f)
            corpus = ["\n".join(s["program"]) for s in train_data]
            tokenizer = SimpleTokenizer()
            tokenizer.train(corpus, verbose=False)
        else:
            raise FileNotFoundError(f"No tokenizer vocab at {vocab_path} and no training data to recreate")

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
