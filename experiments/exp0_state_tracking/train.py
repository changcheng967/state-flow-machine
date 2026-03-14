"""
Training Script for State Tracking Experiment

Trains both SFM Execution System (State Slots) and Transformer baseline
for comparison on the state tracking task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
import time
import os
import sys
import json
import argparse

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig, ExperimentConfig
from sfm.systems.execution import ExecutionSystem
from sfm.utils.device import get_device, set_seed
from baseline_transformer import TransformerEncoderOnly
from dataset import create_dataloaders
from generate_data import generate_and_save
from sfm.tokenizer.code_tokenizer import CodeTokenizer


class StateTrackingWrapper(nn.Module):
    """
    Wraps ExecutionSystem for state tracking classification.
    """

    def __init__(
        self,
        execution_system: ExecutionSystem,
        vocab_size: int,
        num_classes: int = 1000
    ):
        super().__init__()
        self.execution = execution_system

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, execution_system.input_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(execution_system.input_dim),
            nn.Linear(execution_system.input_dim, execution_system.input_dim),
            nn.ReLU(),
            nn.Linear(execution_system.input_dim, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed tokens
        x = self.embedding(input_ids)

        # Process through execution system
        x = self.execution(x)

        # Pool and classify
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Trainer:
    """
    Trainer for state tracking experiment.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        model_name: str = "model",
        is_transformer: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        self.is_transformer = is_transformer

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # Assuming 10 epochs
            eta_min=learning_rate * 0.01
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _get_mask(self, attention_mask):
        """Get mask in the right format for the model type."""
        if self.is_transformer:
            # Transformer expects True for padding positions
            return (attention_mask == 0)
        else:
            # Other models use standard mask
            return attention_mask

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        epoch_start = time.time()
        print(f"  Training started at {time.strftime('%H:%M:%S')}")

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["final_value"].to(self.device)

            # Get mask in correct format for model
            mask = self._get_mask(attention_mask)

            # Forward pass
            self.optimizer.zero_grad()
            if self.is_transformer:
                logits = self.model(input_ids, mask=mask)
            else:
                logits = self.model(input_ids, attention_mask=mask)

            # Clamp labels to valid class range [0, num_classes-1]
            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            self.global_step += 1

            # Progress indicator - print dot for each batch
            sys.stdout.write('.')
            sys.stdout.flush()

            # Print timing for first batch (usually slow due to NPU compilation)
            if batch_idx == 0:
                batch_time = time.time() - batch_start
                print(f' [first batch: {batch_time:.2f}s]')

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        print(f' done [{epoch_time:.2f}s total]')

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "time": epoch_time
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["final_value"].to(self.device)

            # Get mask in correct format for model
            mask = self._get_mask(attention_mask)

            if self.is_transformer:
                logits = self.model(input_ids, mask=mask)
            else:
                logits = self.model(input_ids, attention_mask=mask)

            # Clamp labels to valid class range [0, num_classes-1]
            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }

    def train(
        self,
        num_epochs: int,
        eval_every: int = 100,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of training epochs.
            eval_every: Evaluate every N batches.
            save_dir: Directory to save checkpoints.

        Returns:
            Training history.
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        print(f"\nTraining {self.model_name}...")
        print("-" * 60)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"Time: {train_metrics['time']:.2f}s")

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")

            # Save best model
            if save_dir and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                save_path = os.path.join(save_dir, f"{self.model_name}_best.pt")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "epoch": epoch
                }, save_path)
                print(f"  Saved best model to {save_path}")

        return history


def run_experiment(
    config: ExperimentConfig,
    sfm_config: SFMConfig,
    device: torch.device,
    save_dir: str
) -> Dict:
    """
    Run the full state tracking experiment.

    Args:
        config: Experiment configuration.
        sfm_config: SFM model configuration.
        device: Device to use.
        save_dir: Directory for saving results.

    Returns:
        Results dictionary.
    """
    os.makedirs(save_dir, exist_ok=True)
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate data
    print("Generating data...")
    generate_and_save(
        output_dir=data_dir,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        max_program_length=config.max_program_length,
        seed=config.seed
    )

    # Create tokenizer
    print("\nCreating tokenizer...")
    with open(os.path.join(data_dir, "train.json"), 'r') as f:
        train_data = json.load(f)

    corpus = ["\n".join(s["program"]) for s in train_data]
    tokenizer = CodeTokenizer(vocab_size=5000, min_freq=1)
    tokenizer.train(corpus, verbose=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir,
        tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_program_length * 10
    )

    results = {}

    # Train Execution System (State Slots)
    print("\n" + "=" * 60)
    print("Training Execution System (State Slots)")
    print("=" * 60)

    execution = ExecutionSystem(
        input_dim=sfm_config.d_model,
        hidden_dim=sfm_config.deltanet_hidden_dim,
        num_slots=sfm_config.execution_num_slots,
        slot_dim=sfm_config.execution_slot_dim,
        max_ticks=sfm_config.execution_max_ticks,
        num_heads=sfm_config.execution_num_heads,
        dropout=sfm_config.dropout
    )

    execution_wrapper = StateTrackingWrapper(
        execution,
        vocab_size=tokenizer.vocab_size_actual,
        num_classes=500  # Values 0-500 for classification
    ).to(device)

    print(f"Execution System parameters: {execution_wrapper.count_parameters():,}")

    execution_trainer = Trainer(
        execution_wrapper,
        train_loader,
        val_loader,
        device,
        learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
        model_name="execution"
    )

    # NPU warmup with real batch shape (compiles forward AND backward graphs)
    if "npu" in str(device):
        print("Warming up NPU with real batch shape (Execution System)...")
        warmup_batch = next(iter(train_loader))
        input_ids = warmup_batch["input_ids"].to(device)
        attention_mask = warmup_batch["attention_mask"].to(device)
        labels = warmup_batch["final_value"].to(device).clamp(0, 499)

        # Forward + backward to compile both graphs
        execution_wrapper.train()
        logits = execution_wrapper(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        execution_wrapper.zero_grad()
        print("NPU warmup complete.")

    execution_history = execution_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir
    )
    results["execution"] = execution_history

    # Train Transformer Baseline
    print("\n" + "=" * 60)
    print("Training Transformer Baseline")
    print("=" * 60)

    transformer = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=sfm_config.d_model,
        num_heads=4,
        num_layers=4,
        d_ff=sfm_config.d_model * 4,
        num_output_classes=500
    ).to(device)

    print(f"Transformer parameters: {transformer.count_parameters():,}")

    transformer_trainer = Trainer(
        transformer,
        train_loader,
        val_loader,
        device,
        learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
        model_name="transformer",
        is_transformer=True
    )

    # NPU warmup with real batch shape (compiles forward AND backward graphs)
    if "npu" in str(device):
        print("Warming up NPU with real batch shape (Transformer)...")
        warmup_batch = next(iter(train_loader))
        input_ids = warmup_batch["input_ids"].to(device)
        attention_mask = warmup_batch["attention_mask"].to(device)
        labels = warmup_batch["final_value"].to(device).clamp(0, 499)

        # Forward + backward to compile both graphs
        transformer.train()
        # Transformer expects mask where True = padding
        mask = (attention_mask == 0)
        logits = transformer(input_ids, mask=mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        transformer.zero_grad()
        print("NPU warmup complete.")

    transformer_history = transformer_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir
    )
    results["transformer"] = transformer_history

    # Save results
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train state tracking models")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0", help="Save directory")
    args = parser.parse_args()

    # Setup
    set_seed(42)
    device = get_device()

    if args.quick:
        # Quick test configuration
        exp_config = ExperimentConfig.quick()
        sfm_config = SFMConfig.small()
        exp_config.num_epochs = 1
    else:
        exp_config = ExperimentConfig(
            train_samples=args.samples,
            val_samples=args.samples // 10,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        sfm_config = SFMConfig.small()

    # Run experiment
    results = run_experiment(
        config=exp_config,
        sfm_config=sfm_config,
        device=device,
        save_dir=args.save_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nExecution System (State Slots):")
    print(f"  Final train accuracy: {results['execution']['train_accuracy'][-1]:.4f}")
    print(f"  Final val accuracy: {results['execution']['val_accuracy'][-1]:.4f}")

    print("\nTransformer Baseline:")
    print(f"  Final train accuracy: {results['transformer']['train_accuracy'][-1]:.4f}")
    print(f"  Final val accuracy: {results['transformer']['val_accuracy'][-1]:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
