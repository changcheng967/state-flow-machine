"""
Visualization Script for SFM

Visualizes:
- Training curves
- Generalization comparison
- System parameter breakdown
- State slot activations
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves"
):
    """
    Plot training and validation loss/accuracy.

    Args:
        history: Dict with train_loss, train_accuracy, val_loss, val_accuracy.
        save_path: Path to save figure.
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(epochs, history["train_accuracy"], 'b-', label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], 'r-', label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {save_path}")


def plot_generalization_comparison(
    exec_results: Dict[int, float],
    trans_results: Dict[int, float],
    save_path: str,
    title: str = "Generalization Comparison"
):
    """
    Plot accuracy vs sequence length multiplier for both models.

    Args:
        exec_results: Dict mapping length multiplier to accuracy (Execution).
        trans_results: Dict mapping length multiplier to accuracy (Transformer).
        save_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    multipliers = sorted(exec_results.keys())
    exec_accs = [exec_results[m] for m in multipliers]
    trans_accs = [trans_results[m] for m in multipliers]

    x = np.arange(len(multipliers))
    width = 0.35

    bars1 = ax.bar(x - width/2, exec_accs, width, label='State Slots', color='#2ecc71')
    bars2 = ax.bar(x + width/2, trans_accs, width, label='Transformer', color='#e74c3c')

    ax.set_xlabel('Sequence Length Multiplier')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}x' for m in multipliers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved generalization comparison to {save_path}")


def plot_parameter_breakdown(
    param_counts: Dict[str, int],
    save_path: str,
    title: str = "SFM Parameter Breakdown"
):
    """
    Plot pie chart of parameters per system.

    Args:
        param_counts: Dict mapping system name to parameter count.
        save_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter out small components for clarity
    main_systems = {k: v for k, v in param_counts.items() if v > param_counts.get("total", 1) * 0.01}

    labels = list(main_systems.keys())
    sizes = list(main_systems.values())
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
    colors = colors[:len(labels)]

    # Don't include 'total' in the pie
    if 'total' in labels:
        idx = labels.index('total')
        labels.pop(idx)
        sizes.pop(idx)

    explode = [0.02] * len(labels)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',
        explode=explode,
        startangle=90
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved parameter breakdown to {save_path}")


def plot_system_weights(
    weights: Dict[str, float],
    save_path: str,
    title: str = "SFM System Weights"
):
    """
    Plot bar chart of system weights.

    Args:
        weights: Dict mapping system name to weight.
        save_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(weights.keys())
    values = list(weights.values())
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

    bars = ax.bar(names, values, color=colors[:len(names)])

    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved system weights to {save_path}")


def visualize_experiment_results(
    results_dir: str,
    output_dir: str
):
    """
    Visualize all results from an experiment directory.

    Args:
        results_dir: Directory containing experiment results.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results_path = os.path.join(results_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Plot training curves for each model
        for model_name in ["execution", "transformer"]:
            if model_name in results:
                history = results[model_name]
                plot_training_curves(
                    history,
                    os.path.join(output_dir, f"{model_name}_training.png"),
                    title=f"{model_name.capitalize()} Training Curves"
                )

    # Load evaluation results
    eval_path = os.path.join(results_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)

        # Convert string keys to int for multipliers
        exec_results = {int(k): v["accuracy"] for k, v in eval_results.get("execution", {}).items()}
        trans_results = {int(k): v["accuracy"] for k, v in eval_results.get("transformer", {}).items()}

        if exec_results and trans_results:
            plot_generalization_comparison(
                exec_results,
                trans_results,
                os.path.join(output_dir, "generalization_comparison.png"),
                title="State Tracking: Generalization to Longer Sequences"
            )


def visualize_model(
    model_path: str,
    output_dir: str
):
    """
    Visualize model architecture and parameters.

    Args:
        model_path: Path to model checkpoint.
        output_dir: Directory to save visualizations.
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract parameter counts
    param_counts = {}
    for name, param in checkpoint.get("model_state_dict", {}).items():
        # Group by system
        parts = name.split('.')
        if len(parts) > 1:
            system = parts[0]
            if system not in param_counts:
                param_counts[system] = 0
            param_counts[system] += param.numel()

    if param_counts:
        plot_parameter_breakdown(
            param_counts,
            os.path.join(output_dir, "parameter_breakdown.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SFM experiment results")
    parser.add_argument("--results_dir", type=str, default="outputs/exp0",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations (default: results_dir/figures)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint for architecture visualization")

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "figures")

    # Visualize experiment results
    visualize_experiment_results(args.results_dir, output_dir)

    # Visualize model if provided
    if args.model_path:
        visualize_model(args.model_path, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")
