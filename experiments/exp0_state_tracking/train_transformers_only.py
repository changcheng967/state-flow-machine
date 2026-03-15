"""
Standalone Transformer Training + Evaluation Script

Trains Transformer-Fair and Transformer-Large baselines on a single NPU,
then runs evaluation. No DDP, no distributed training.

Usage:
    python experiments/exp0_state_tracking/train_transformers_only.py

Requires existing tokenizer at outputs/exp0/tokenizer_vocab.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Optional
import time
import sys
import os
import subprocess
import argparse

# IMPORTANT: Import torch_npu FIRST
import torch_npu

# Apply NPU optimizations
os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '1200')
os.environ.setdefault('HCCL_EXEC_TIMEOUT', '1200')

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig, ExperimentConfig
from sfm.utils.device import get_device, set_seed, synchronize
from sfm.tokenizer.code_tokenizer import SimpleTokenizer
from baseline_transformer import TransformerEncoderOnly
from dataset import create_dataloaders
from generate_data import generate_and_save


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(3.14159265359 * progress)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """Trainer with AMP and cosine warmup scheduler."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        model_name: str = "model",
        is_transformer: bool = False,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        warmup_steps: int = 500,
        num_epochs: int = 50
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        self.is_transformer = is_transformer
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        total_steps = len(train_loader) * num_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-6
        )

        self.criterion = nn.MSELoss()

        if use_amp:
            try:
                self.scaler = torch.npu.amp.GradScaler(
                    init_scale=1024.0,
                    growth_factor=1.5,
                    backoff_factor=0.5,
                    growth_interval=500
                )
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=1024.0,
                    growth_factor=1.5,
                    backoff_factor=0.5,
                    growth_interval=500
                )
        else:
            self.scaler = None

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _get_mask(self, attention_mask):
        """Get mask in the right format for the model type."""
        if self.is_transformer:
            return (attention_mask == 0)
        return attention_mask

    def _forward_with_amp(self, input_ids, mask):
        """Forward pass with AMP support."""
        if self.use_amp:
            try:
                with torch.npu.amp.autocast():
                    if self.is_transformer:
                        return self.model(input_ids, mask=mask)
                    else:
                        return self.model(input_ids, attention_mask=mask)
            except AttributeError:
                with torch.cuda.amp.autocast():
                    if self.is_transformer:
                        return self.model(input_ids, mask=mask)
                    else:
                        return self.model(input_ids, attention_mask=mask)
        else:
            if self.is_transformer:
                return self.model(input_ids, mask=mask)
            else:
                return self.model(input_ids, attention_mask=mask)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        epoch_start = time.time()
        print(f"  Training started, LR: {self.scheduler.get_lr():.2e}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels_norm = batch["final_value_normalized"].to(self.device).float()
            labels_raw = batch["final_value"].to(self.device)

            mask = self._get_mask(attention_mask)

            predictions = self._forward_with_amp(input_ids, mask)

            # Loss in FP32 outside autocast to avoid dtype mismatch
            loss = self.criterion(predictions.float(), labels_norm.float())

            loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            predicted_values = (predictions * 100).round().clamp(0, 100)
            total_correct += (predicted_values == labels_raw).sum().item()
            total_samples += labels_raw.size(0)
            self.global_step += 1

            if batch_idx % 50 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        synchronize()
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        print(f' done [{epoch_time:.2f}s total]')

        return {"loss": avg_loss, "accuracy": accuracy, "time": epoch_time}

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
            labels_norm = batch["final_value_normalized"].to(self.device).float()
            labels_raw = batch["final_value"].to(self.device)

            mask = self._get_mask(attention_mask)

            if self.is_transformer:
                predictions = self.model(input_ids, mask=mask)
            else:
                predictions = self.model(input_ids, attention_mask=mask)

            # Loss in FP32 to avoid dtype mismatch
            loss = self.criterion(predictions.float(), labels_norm.float())

            total_loss += loss.item()
            predicted_values = (predictions * 100).round().clamp(0, 100)
            total_correct += (predicted_values == labels_raw).sum().item()
            total_samples += labels_raw.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    def train(self, num_epochs: int, save_dir: Optional[str] = None) -> Dict[str, list]:
        """Train model for multiple epochs."""
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

        print(f"\nTraining {self.model_name}...")
        print("-" * 60)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")

            synchronize()
            val_metrics = self.evaluate(self.val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

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


def warmup_npu(model, train_loader, device, is_transformer: bool = False, use_amp: bool = True):
    """Warmup NPU with real batch to compile graphs."""
    print("Warming up NPU with real batch shape...")
    synchronize()

    warmup_batch = next(iter(train_loader))
    input_ids = warmup_batch["input_ids"].to(device)
    attention_mask = warmup_batch["attention_mask"].to(device)
    labels_norm = warmup_batch["final_value_normalized"].to(device).float()

    model.train()

    if use_amp:
        try:
            with torch.npu.amp.autocast():
                if is_transformer:
                    mask = (attention_mask == 0)
                    predictions = model(input_ids, mask=mask)
                else:
                    predictions = model(input_ids, attention_mask=attention_mask)
        except AttributeError:
            with torch.cuda.amp.autocast():
                if is_transformer:
                    mask = (attention_mask == 0)
                    predictions = model(input_ids, mask=mask)
                else:
                    predictions = model(input_ids, attention_mask=attention_mask)
    else:
        if is_transformer:
            mask = (attention_mask == 0)
            predictions = model(input_ids, mask=mask)
        else:
            predictions = model(input_ids, attention_mask=attention_mask)

    # Loss in FP32 outside autocast to avoid dtype mismatch
    loss = F.mse_loss(predictions.float(), labels_norm.float())
    loss.backward()
    model.zero_grad(set_to_none=True)

    synchronize()
    print("NPU warmup complete (graph compiled).")


def main():
    parser = argparse.ArgumentParser(description="Train transformers only (standalone)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Validation samples")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0", help="Save directory")
    parser.add_argument("--difficulty", type=str, default="hard",
                        choices=["easy", "medium", "hard"], help="Difficulty level")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    use_amp = "npu" in str(device)

    print("=" * 70)
    print("STANDALONE TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Save directory: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Check for existing tokenizer
    vocab_path = os.path.join(args.save_dir, "tokenizer_vocab.json")
    if not os.path.exists(vocab_path):
        print(f"\n[ERROR] Tokenizer not found at {vocab_path}")
        print("Run the main training script first to generate tokenizer.")
        return 1

    # Load tokenizer
    print(f"\nLoading tokenizer from {vocab_path}...")
    tokenizer = SimpleTokenizer.load(vocab_path)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size_actual}")

    # Check for existing data or generate
    data_dir = os.path.join(args.save_dir, "data")
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"\nGenerating data to {data_dir}...")
        generate_and_save(
            output_dir=data_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            max_program_length=20,
            seed=42,
            difficulty=args.difficulty
        )
    else:
        print(f"\nUsing existing data from {data_dir}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir,
        tokenizer,
        batch_size=args.batch_size,
        max_length=200,
        distributed=False,
        rank=0,
        world_size=1
    )

    sfm_config = SFMConfig.small()
    timings = {}
    results = {}

    # ========== Train Transformer-Fair ==========
    print("\n" + "=" * 60)
    print("Training Transformer-Fair Baseline")
    print("=" * 60)

    t0 = time.time()
    transformer_fair = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        num_output_classes=1
    ).to(device)

    trans_fair_params = sum(p.numel() for p in transformer_fair.parameters())
    print(f"Transformer-Fair parameters: {trans_fair_params:,}")

    if use_amp:
        warmup_npu(transformer_fair, train_loader, device, is_transformer=True, use_amp=True)
    timings["transformer_fair_warmup"] = time.time() - t0

    trainer_fair = Trainer(
        transformer_fair,
        train_loader,
        val_loader,
        device,
        learning_rate=3e-4,
        model_name="transformer_fair",
        is_transformer=True,
        grad_accum_steps=1,
        use_amp=use_amp,
        warmup_steps=500,
        num_epochs=args.epochs
    )

    t0 = time.time()
    synchronize()
    history_fair = trainer_fair.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    synchronize()
    timings["transformer_fair_training"] = time.time() - t0
    results["transformer_fair"] = history_fair

    print(f"\nTransformer-Fair Final Results:")
    print(f"  Train Loss: {history_fair['train_loss'][-1]:.4f}, Accuracy: {history_fair['train_accuracy'][-1]:.4f}")
    print(f"  Val Loss: {history_fair['val_loss'][-1]:.4f}, Accuracy: {history_fair['val_accuracy'][-1]:.4f}")
    print(f"  Training time: {timings['transformer_fair_training']:.1f}s")

    # ========== Train Transformer-Large ==========
    print("\n" + "=" * 60)
    print("Training Transformer-Large Baseline")
    print("=" * 60)

    t0 = time.time()
    transformer_large = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        num_output_classes=1
    ).to(device)

    trans_large_params = sum(p.numel() for p in transformer_large.parameters())
    print(f"Transformer-Large parameters: {trans_large_params:,}")

    if use_amp:
        warmup_npu(transformer_large, train_loader, device, is_transformer=True, use_amp=True)
    timings["transformer_large_warmup"] = time.time() - t0

    trainer_large = Trainer(
        transformer_large,
        train_loader,
        val_loader,
        device,
        learning_rate=3e-4,
        model_name="transformer_large",
        is_transformer=True,
        grad_accum_steps=1,
        use_amp=use_amp,
        warmup_steps=500,
        num_epochs=args.epochs
    )

    t0 = time.time()
    synchronize()
    history_large = trainer_large.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    synchronize()
    timings["transformer_large_training"] = time.time() - t0
    results["transformer_large"] = history_large

    print(f"\nTransformer-Large Final Results:")
    print(f"  Train Loss: {history_large['train_loss'][-1]:.4f}, Accuracy: {history_large['train_accuracy'][-1]:.4f}")
    print(f"  Val Loss: {history_large['val_loss'][-1]:.4f}, Accuracy: {history_large['val_accuracy'][-1]:.4f}")
    print(f"  Training time: {timings['transformer_large_training']:.1f}s")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\nParameter Counts:")
    print(f"  Transformer-Fair:  {trans_fair_params:,}")
    print(f"  Transformer-Large: {trans_large_params:,}")

    total_time = sum(timings.values())
    print(f"\nTiming Breakdown:")
    for name, t in timings.items():
        print(f"  {name}: {t:.1f}s ({t/total_time*100:.1f}%)")
    print(f"  TOTAL: {total_time:.1f}s")

    print(f"\nCheckpoints saved to:")
    print(f"  {os.path.join(args.save_dir, 'transformer_fair_best.pt')}")
    print(f"  {os.path.join(args.save_dir, 'transformer_large_best.pt')}")

    # ========== Run Evaluation ==========
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)

        eval_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
        eval_cmd = [
            sys.executable, eval_script,
            "--save_dir", args.save_dir,
            "--difficulty", args.difficulty,
            "--base_length", "10",
            "--multipliers", "1", "2", "4", "8",
            "--samples", "1000"
        ]

        print(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"\n[WARNING] Evaluation returned exit code {result.returncode}")
        else:
            print("\nEvaluation completed successfully!")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
