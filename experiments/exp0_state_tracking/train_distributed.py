"""
Distributed Training Script for State Tracking Experiment

Supports:
- Single NPU training
- Multi-NPU DataParallel
- Multi-NPU DistributedDataParallel (DDP)
- Mixed precision training
- Gradient accumulation

Usage:
    # Single NPU
    python train_distributed.py

    # Multi-NPU with torchrun
    torchrun --nproc_per_node=4 train_distributed.py

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
        --master_addr=10.0.0.1 --master_port=29500 train_distributed.py
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sfm.config import SFMConfig, ExperimentConfig
from sfm.systems.execution import ExecutionSystem
from sfm.utils.device import get_device
from sfm.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model,
    create_distributed_dataloader,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    all_reduce_tensor,
    barrier,
    GradientAccumulator,
    MixedPrecisionTrainer,
    get_effective_batch_size,
)
from dataset import StateTrackingDataset
from generate_data import SimpleProgramGenerator
from sfm.tokenizer.code_tokenizer import CodeTokenizer


class StateTrackingWrapper(nn.Module):
    """Wraps ExecutionSystem for state tracking classification."""

    def __init__(
        self,
        execution_system: ExecutionSystem,
        vocab_size: int,
        num_classes: int = 1000
    ):
        super().__init__()
        self.execution = execution_system
        self.embedding = nn.Embedding(vocab_size, execution_system.input_dim)
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
        x = self.embedding(input_ids)
        x = self.execution(x)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled)


class DistributedTrainer:
    """
    Distributed trainer for state tracking experiment.

    Supports:
    - DDP across multiple NPUs
    - Gradient accumulation
    - Mixed precision training
    - Distributed evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        model_name: str = "model"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Get actual model for optimizer (unwrap DDP)
        self.raw_model = model.module if hasattr(model, 'module') else model

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        num_training_steps = len(train_loader) * 10 // gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=learning_rate * 0.01
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.amp_trainer = MixedPrecisionTrainer(enabled=use_amp)

        # Gradient accumulator
        self.grad_accumulator = GradientAccumulator(gradient_accumulation_steps)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)

        start_time = time.time()

        # Set epoch for distributed sampler
        if is_distributed() and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.global_step)

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["final_value"].to(self.device)

            # Forward pass with AMP
            logits = self.amp_trainer.forward(self.model, input_ids, attention_mask)

            # Clamp labels
            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            # Compute loss (scale for gradient accumulation)
            loss = self.criterion(logits, labels)
            loss = loss / self.gradient_accumulation_steps

            # Backward pass with AMP
            self.amp_trainer.backward(
                loss,
                self.optimizer,
                clip_grad_norm=self.max_grad_norm,
                model=self.raw_model
            )

            # Gradient accumulation
            if self.grad_accumulator.step():
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().float()
            total_samples += labels.size(0)

            self.global_step += 1

        # Aggregate across all processes
        total_loss = all_reduce_tensor(total_loss, average=True)
        total_correct = all_reduce_tensor(total_correct, average=True)
        total_samples = all_reduce_tensor(total_samples, average=True)

        epoch_time = time.time() - start_time
        avg_loss = total_loss.item() / len(self.train_loader)
        accuracy = total_correct.item() / total_samples.item() if total_samples > 0 else 0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "time": epoch_time
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["final_value"].to(self.device)

            logits = self.amp_trainer.forward(self.model, input_ids, attention_mask)

            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().float()
            total_samples += labels.size(0)

        # Aggregate across all processes
        total_loss = all_reduce_tensor(total_loss, average=True)
        total_correct = all_reduce_tensor(total_correct, average=True)
        total_samples = all_reduce_tensor(total_samples, average=True)

        avg_loss = total_loss.item() / len(dataloader)
        accuracy = total_correct.item() / total_samples.item() if total_samples > 0 else 0

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }

    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """Train model for multiple epochs."""
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        if is_main_process():
            print(f"\nTraining {self.model_name}...")
            print("-" * 60)

        for epoch in range(num_epochs):
            if is_main_process():
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()

            if is_main_process():
                history["train_loss"].append(train_metrics["loss"])
                history["train_accuracy"].append(train_metrics["accuracy"])
                print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                      f"Accuracy: {train_metrics['accuracy']:.4f}, "
                      f"Time: {train_metrics['time']:.2f}s")

            # Synchronize before evaluation
            barrier()

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            if is_main_process():
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Accuracy: {val_metrics['accuracy']:.4f}")

                # Save best model (only on main process)
                if save_dir and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{self.model_name}_best.pt")
                    torch.save({
                        "model_state_dict": self.raw_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "epoch": epoch
                    }, save_path)
                    print(f"  Saved best model to {save_path}")

            barrier()

        return history


def run_distributed_experiment(
    config: ExperimentConfig,
    sfm_config: SFMConfig,
    save_dir: str,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True
) -> Dict:
    """
    Run distributed training experiment.

    Args:
        config: Experiment configuration.
        sfm_config: SFM model configuration.
        save_dir: Directory for saving results.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_amp: Whether to use mixed precision.

    Returns:
        Training results.
    """
    # Setup distributed training
    rank, world_size, device = setup_distributed()

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Generate data (only on main process)
        print("Generating data...")
        generator = SimpleProgramGenerator(
            num_variables=5,
            max_program_length=config.max_program_length,
            seed=config.seed
        )

        train_data = generator.generate_dataset(config.train_samples, include_trace=False)
        val_data = generator.generate_dataset(config.val_samples, include_trace=False)

        train_path = os.path.join(data_dir, "train.json")
        val_path = os.path.join(data_dir, "val.json")

        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(val_path, 'w') as f:
            json.dump(val_data, f)

        print(f"Saved training data to {train_path}")
        print(f"Saved validation data to {val_path}")

        # Create tokenizer
        print("\nCreating tokenizer...")
        corpus = ["\n".join(s["program"]) for s in train_data + val_data]
        tokenizer = CodeTokenizer(vocab_size=5000, min_freq=1)
        tokenizer.train(corpus, verbose=True)

    # Synchronize before creating datasets
    barrier()

    if not is_main_process():
        # Load tokenizer on other processes
        data_dir = os.path.join(save_dir, "data")
        train_path = os.path.join(data_dir, "train.json")
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(os.path.join(data_dir, "val.json"), 'r') as f:
            val_data = json.load(f)

        corpus = ["\n".join(s["program"]) for s in train_data + val_data]
        tokenizer = CodeTokenizer(vocab_size=5000, min_freq=1)
        tokenizer.train(corpus, verbose=False)

    # Create datasets
    train_dataset = StateTrackingDataset(
        os.path.join(save_dir, "data", "train.json"),
        tokenizer,
        max_length=config.max_program_length * 10
    )
    val_dataset = StateTrackingDataset(
        os.path.join(save_dir, "data", "val.json"),
        tokenizer,
        max_length=config.max_program_length * 10
    )

    # Create distributed dataloaders
    train_loader = create_distributed_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = create_distributed_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )

    if is_main_process():
        print(f"\nDataset statistics:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  World size: {world_size} NPUs")
        print(f"  Effective batch size: {get_effective_batch_size(config.batch_size, world_size, gradient_accumulation_steps)}")

    results = {}

    # Train Execution System
    if is_main_process():
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
        num_classes=500
    )

    # Wrap with DDP if distributed
    execution_wrapper = wrap_model(execution_wrapper, device, distributed=is_distributed())

    if is_main_process():
        print(f"Execution System parameters: {sum(p.numel() for p in execution.parameters()):,}")

    trainer = DistributedTrainer(
        execution_wrapper,
        train_loader,
        val_loader,
        device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp,
        model_name="execution"
    )

    execution_history = trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if is_main_process() else None
    )
    results["execution"] = execution_history

    # Save results
    if is_main_process():
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Distributed training for state tracking")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-NPU batch size")
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0", help="Save directory")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    if args.quick:
        exp_config = ExperimentConfig.quick()
        exp_config.num_epochs = 1
    else:
        exp_config = ExperimentConfig(
            train_samples=args.samples,
            val_samples=args.samples // 10,
            max_program_length=20,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )

    sfm_config = SFMConfig.small()

    try:
        results = run_distributed_experiment(
            config=exp_config,
            sfm_config=sfm_config,
            save_dir=args.save_dir,
            gradient_accumulation_steps=args.grad_accum,
            use_amp=not args.no_amp
        )

        if is_main_process():
            print("\n" + "=" * 60)
            print("Training complete!")
            print("=" * 60)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
