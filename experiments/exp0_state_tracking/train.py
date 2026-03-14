"""
Training Script for State Tracking Experiment

Trains both SFM Execution System (State Slots) and Transformer baseline
for comparison on the state tracking task.

CUBE-OPTIMIZED for Ascend NPU:
- Multi-NPU DDP support (HCCL backend)
- Automatic Mixed Precision (AMP) for FP16 training
- Gradient accumulation: effective_batch = batch_size * npus * accumulation_steps
- Default: batch_size=32, accumulation_steps=2, so effective_batch = 32*4*2 = 256
- Cosine annealing with linear warmup (500 steps)
- torch.npu.synchronize() before timing measurements
- Warmup with REAL batch shape

LEARNING FIXES:
- REGRESSION with MSE loss (not 500-class classification)
- Output head: nn.Linear(d_model, 1) with sigmoid
- Labels: float values normalized to [0, 1]
- Accuracy = (prediction * 100).round() == target
- Learning rate: 3e-4
- Epochs: 50
"""

# IMPORTANT: Import torch_npu FIRST before any torch.npu calls
import torch_npu

# Apply NPU optimizations
import os
os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '1200')
os.environ.setdefault('HCCL_EXEC_TIMEOUT', '1200')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from typing import Dict, Optional, Tuple
import time
import sys
import json
import argparse
import math

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig, ExperimentConfig
from sfm.systems.execution import ExecutionSystem
from sfm.utils.device import get_device, set_seed, synchronize
from sfm.tokenizer.code_tokenizer import SimpleTokenizer
from baseline_transformer import TransformerEncoderOnly
from dataset import create_dataloaders
from generate_data import generate_and_save


class WarmupCosineScheduler:
    """
    Cosine annealing with linear warmup.

    - Linear warmup for warmup_steps
    - Then cosine decay to min_lr
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
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
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class StateTrackingWrapper(nn.Module):
    """Wraps ExecutionSystem for state tracking REGRESSION."""

    def __init__(
        self,
        execution_system: ExecutionSystem,
        vocab_size: int,
        num_classes: int = 1  # REGRESSION: single scalar output
    ):
        super().__init__()
        self.execution = execution_system
        self.embedding = nn.Embedding(vocab_size, execution_system.input_dim)
        # REGRESSION: Output single scalar, apply sigmoid to constrain to [0, 1]
        self.regressor = nn.Sequential(
            nn.LayerNorm(execution_system.input_dim),
            nn.Linear(execution_system.input_dim, execution_system.input_dim),
            nn.ReLU(),
            nn.Linear(execution_system.input_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Trainer:
    """Trainer with AMP, gradient accumulation, and cosine warmup scheduler."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 3e-4,  # INCREASED from 1e-4
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        model_name: str = "model",
        is_transformer: bool = False,
        grad_accum_steps: int = 2,  # Default for effective batch 256
        use_amp: bool = True,
        rank: int = 0,
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
        self.rank = rank

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Cosine annealing with warmup
        total_steps = len(train_loader) * num_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-6
        )

        # REGRESSION: Use MSELoss
        self.criterion = nn.MSELoss()

        # AMP scaler with more conservative settings
        if use_amp:
            try:
                self.scaler = torch.npu.amp.GradScaler(
                    init_scale=1024.0,  # Start lower
                    growth_factor=1.5,  # Grow slower
                    backoff_factor=0.5,  # Back off faster
                    growth_interval=500  # Check less frequently
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
            return (attention_mask == 0)  # True for padding
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
        """Train for one epoch with gradient accumulation and AMP (REGRESSION)."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        epoch_start = time.time()
        if self.rank == 0:
            print(f"  Training started at {time.strftime('%H:%M:%S')}, LR: {self.scheduler.get_lr():.2e}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels_norm = batch["final_value_normalized"].to(self.device).float()
            labels_raw = batch["final_value"].to(self.device)

            mask = self._get_mask(attention_mask)

            # Forward with AMP
            predictions = self._forward_with_amp(input_ids, mask)

            # Compute MSE loss
            if self.use_amp:
                try:
                    with torch.npu.amp.autocast():
                        loss = self.criterion(predictions, labels_norm)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        loss = self.criterion(predictions, labels_norm)
            else:
                loss = self.criterion(predictions, labels_norm)

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

            # Backward with AMP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every grad_accum_steps
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

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            predicted_values = (predictions * 100).round().clamp(0, 100)
            total_correct += (predicted_values == labels_raw).sum().item()
            total_samples += labels_raw.size(0)
            self.global_step += 1

            # Progress indicator
            if self.rank == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
                if batch_idx == 0:
                    synchronize()  # Ensure accurate timing
                    batch_time = time.time() - epoch_start
                    print(f' [first batch: {batch_time:.2f}s]')

        synchronize()  # Synchronize before timing
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        if self.rank == 0:
            print(f' done [{epoch_time:.2f}s total]')

        return {"loss": avg_loss, "accuracy": accuracy, "time": epoch_time}

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model (REGRESSION)."""
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

            loss = self.criterion(predictions, labels_norm)

            total_loss += loss.item()
            predicted_values = (predictions * 100).round().clamp(0, 100)
            total_correct += (predicted_values == labels_raw).sum().item()
            total_samples += labels_raw.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return {"loss": avg_loss, "accuracy": accuracy}

    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """Train model for multiple epochs."""
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

        if self.rank == 0:
            print(f"\nTraining {self.model_name}...")
            print(f"  Effective batch size: {len(self.train_loader.dataset) // len(self.train_loader) * self.grad_accum_steps}")
            print("-" * 60)

        for epoch in range(num_epochs):
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            if self.rank == 0:
                print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                      f"Accuracy: {train_metrics['accuracy']:.4f}, "
                      f"Time: {train_metrics['time']:.2f}s")

            # Validate (only rank 0)
            if self.rank == 0:
                if dist.is_initialized():
                    dist.barrier()
                synchronize()
                val_metrics = self.evaluate(self.val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Accuracy: {val_metrics['accuracy']:.4f}")

                # Save best model
                if save_dir and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    save_path = os.path.join(save_dir, f"{self.model_name}_best.pt")
                    torch.save({
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "epoch": epoch
                    }, save_path)
                    print(f"  Saved best model to {save_path}")

        return history


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training with HCCL backend for Ascend NPUs."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    backend = 'hccl'
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Initialized distributed training with backend: {backend}")
    except Exception as e:
        print(f"[Rank {rank}] ERROR: HCCL backend initialization failed: {e}")
        print(f"[Rank {rank}] HCCL is required for multi-NPU training on Ascend. Exiting.")
        sys.exit(1)

    torch.npu.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def warmup_npu(model, train_loader, device, is_transformer: bool = False):
    """Warmup NPU with real batch to compile graphs (REGRESSION)."""
    print("Warming up NPU with real batch shape...")
    synchronize()

    warmup_batch = next(iter(train_loader))
    input_ids = warmup_batch["input_ids"].to(device)
    attention_mask = warmup_batch["attention_mask"].to(device)
    labels_norm = warmup_batch["final_value_normalized"].to(device).float()

    model.train()

    # Forward
    if is_transformer:
        mask = (attention_mask == 0)
        predictions = model(input_ids, mask=mask)
    else:
        predictions = model(input_ids, attention_mask=attention_mask)

    # Backward with MSE loss
    loss = F.mse_loss(predictions, labels_norm)
    loss.backward()
    model.zero_grad(set_to_none=True)

    synchronize()
    print("NPU warmup complete (graph compiled).")


def run_experiment(
    config: ExperimentConfig,
    sfm_config: SFMConfig,
    device: torch.device,
    save_dir: str,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Dict:
    """Run the full state tracking experiment."""
    timings = {}
    exec_params = 0
    trans_params = 0

    # Rank 0 creates directories and generates data
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        print("Generating data...")
        t0 = time.time()
        generate_and_save(
            output_dir=data_dir,
            train_samples=config.train_samples,
            val_samples=config.val_samples,
            max_program_length=config.max_program_length,
            seed=config.seed,
            difficulty=config.difficulty
        )
        timings["data_generation"] = time.time() - t0
    else:
        data_dir = os.path.join(save_dir, "data")

    if distributed:
        dist.barrier()

    # Create tokenizer
    if rank == 0:
        print("\nCreating tokenizer...")
    t0 = time.time()
    with open(os.path.join(data_dir, "train.json"), 'r') as f:
        train_data = json.load(f)
    corpus = ["\n".join(s["program"]) for s in train_data]
    tokenizer = SimpleTokenizer()
    tokenizer.train(corpus, verbose=(rank == 0))
    vocab_path = os.path.join(save_dir, "tokenizer_vocab.json")
    if rank == 0:
        tokenizer.save(vocab_path)
        print(f"Saved tokenizer vocabulary to {vocab_path}")
    timings["tokenization"] = time.time() - t0

    # Create dataloaders
    if rank == 0:
        print("\nCreating dataloaders...")
    t0 = time.time()
    train_loader, val_loader = create_dataloaders(
        data_dir,
        tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_program_length * 10,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    timings["dataloader_creation"] = time.time() - t0

    results = {}
    # Gradient accumulation: effective_batch = batch_size * world_size * grad_accum_steps
    # For 32 * 4 * 2 = 256
    grad_accum_steps = config.grad_accum_steps if distributed else 1
    use_amp = "npu" in str(device)

    # ========== Train Execution System ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Execution System (State Slots)")
        print("=" * 60)

    t0 = time.time()
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
        num_classes=1
    ).to(device)

    if distributed:
        execution_wrapper = DDP(execution_wrapper, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        model_for_count = execution_wrapper.module if hasattr(execution_wrapper, 'module') else execution_wrapper
        exec_params = model_for_count.count_parameters()
        print(f"Execution System parameters: {exec_params:,}")

    if "npu" in str(device):
        model_to_warmup = execution_wrapper.module if hasattr(execution_wrapper, 'module') else execution_wrapper
        warmup_npu(model_to_warmup, train_loader, device, is_transformer=False)
    timings["model_creation_and_warmup"] = time.time() - t0

    execution_trainer = Trainer(
        execution_wrapper,
        train_loader,
        val_loader,
        device,
        learning_rate=sfm_config.learning_rate,
        model_name="execution",
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        rank=rank,
        warmup_steps=sfm_config.warmup_steps,
        num_epochs=config.num_epochs
    )

    t0 = time.time()
    synchronize()
    execution_history = execution_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if rank == 0 else None
    )
    synchronize()
    timings["execution_training"] = time.time() - t0
    results["execution"] = execution_history

    # ========== Train Transformer-Fair Baseline (~660K params to match State Slots) ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Transformer-Fair Baseline (~660K params)")
        print("=" * 60)

    t0 = time.time()
    transformer_fair = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=160,           # Smaller for fair comparison
        num_heads=4,
        num_layers=3,
        d_ff=640,            # d_model * 4
        num_output_classes=1
    ).to(device)

    if distributed:
        transformer_fair = DDP(transformer_fair, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        model_for_count = transformer_fair.module if hasattr(transformer_fair, 'module') else transformer_fair
        trans_fair_params = model_for_count.count_parameters()
        print(f"Transformer-Fair parameters: {trans_fair_params:,}")

    if "npu" in str(device):
        model_to_warmup = transformer_fair.module if hasattr(transformer_fair, 'module') else transformer_fair
        warmup_npu(model_to_warmup, train_loader, device, is_transformer=True)
    timings["transformer_fair_creation_and_warmup"] = time.time() - t0

    transformer_fair_trainer = Trainer(
        transformer_fair,
        train_loader,
        val_loader,
        device,
        learning_rate=sfm_config.learning_rate,
        model_name="transformer_fair",
        is_transformer=True,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        rank=rank,
        warmup_steps=sfm_config.warmup_steps,
        num_epochs=config.num_epochs
    )

    t0 = time.time()
    synchronize()
    transformer_fair_history = transformer_fair_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if rank == 0 else None
    )
    synchronize()
    timings["transformer_fair_training"] = time.time() - t0
    results["transformer_fair"] = transformer_fair_history

    trans_fair_params = trans_fair_params  # Store for later

    # ========== Train Transformer-Large Baseline (~3.26M params, for reference) ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Transformer-Large Baseline (~3.26M params)")
        print("=" * 60)

    t0 = time.time()
    transformer_large = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=sfm_config.d_model,  # Original 256
        num_heads=4,
        num_layers=4,
        d_ff=sfm_config.d_model * 4,  # Original 1024
        num_output_classes=1
    ).to(device)

    if distributed:
        transformer_large = DDP(transformer_large, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        model_for_count = transformer_large.module if hasattr(transformer_large, 'module') else transformer_large
        trans_large_params = model_for_count.count_parameters()
        print(f"Transformer-Large parameters: {trans_large_params:,}")

    if "npu" in str(device):
        model_to_warmup = transformer_large.module if hasattr(transformer_large, 'module') else transformer_large
        warmup_npu(model_to_warmup, train_loader, device, is_transformer=True)
    timings["transformer_large_creation_and_warmup"] = time.time() - t0

    transformer_large_trainer = Trainer(
        transformer_large,
        train_loader,
        val_loader,
        device,
        learning_rate=sfm_config.learning_rate,
        model_name="transformer_large",
        is_transformer=True,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        rank=rank,
        warmup_steps=sfm_config.warmup_steps,
        num_epochs=config.num_epochs
    )

    t0 = time.time()
    synchronize()
    transformer_large_history = transformer_large_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if rank == 0 else None
    )
    synchronize()
    timings["transformer_large_training"] = time.time() - t0
    results["transformer_large"] = transformer_large_history

    # Save results
    if rank == 0:
        results["timings"] = timings
        # Save parameter counts for baseline fairness verification
        results["parameter_counts"] = {
            "state_slots": exec_params,
            "transformer_fair": trans_fair_params,
            "transformer_large": trans_large_params,
            "ratio_fair": round(trans_fair_params / exec_params, 2) if exec_params > 0 else 0,
            "ratio_large": round(trans_large_params / exec_params, 2) if exec_params > 0 else 0
        }
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Print parameter comparison for baseline fairness
        print("\n" + "=" * 60)
        print("PARAMETER COUNTS (Baseline Fairness Check)")
        print("=" * 60)
        print(f"  State Slots:       {exec_params:,} params")
        print(f"  Transformer-Fair:  {trans_fair_params:,} params (ratio: {results['parameter_counts']['ratio_fair']}x)")
        print(f"  Transformer-Large: {trans_large_params:,} params (ratio: {results['parameter_counts']['ratio_large']}x)")
        if 0.8 <= results['parameter_counts']['ratio_fair'] <= 1.25:
            print("  [OK] Transformer-Fair is approximately parameter-matched with State Slots")
        else:
            print("  [!] WARNING: Transformer-Fair may not be fairly matched")

        print("\n" + "=" * 60)
        print("TIMING BREAKDOWN")
        print("=" * 60)
        total_time = sum(timings.values())
        for name, t in timings.items():
            print(f"  {name}: {t:.2f}s ({t/total_time*100:.1f}%)")
        print(f"  TOTAL: {total_time:.2f}s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train state tracking models")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0", help="Save directory")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"], help="Difficulty level")
    args = parser.parse_args()

    # Get local_rank from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        setup_distributed(local_rank, world_size)
        device = torch.device(f"npu:{local_rank}")

        if args.quick:
            exp_config = ExperimentConfig.quick()
            exp_config.difficulty = args.difficulty
            sfm_config = SFMConfig.small()
            exp_config.num_epochs = 1
        else:
            exp_config = ExperimentConfig(
                train_samples=args.samples,
                val_samples=args.samples // 10,
                max_program_length=20,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                difficulty=args.difficulty
            )
            sfm_config = SFMConfig.small()

        try:
            run_experiment(
                config=exp_config,
                sfm_config=sfm_config,
                device=device,
                save_dir=args.save_dir,
                distributed=True,
                rank=local_rank,
                world_size=world_size
            )
        finally:
            cleanup_distributed()
    else:
        set_seed(42)
        device = get_device()

        if args.quick:
            exp_config = ExperimentConfig.quick()
            exp_config.difficulty = args.difficulty
            sfm_config = SFMConfig.small()
            exp_config.num_epochs = 1
        else:
            exp_config = ExperimentConfig(
                train_samples=args.samples,
                val_samples=args.samples // 10,
                max_program_length=20,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                difficulty=args.difficulty
            )
            sfm_config = SFMConfig.small()

        results = run_experiment(
            config=exp_config,
            sfm_config=sfm_config,
            device=device,
            save_dir=args.save_dir,
            distributed=False,
            rank=0,
            world_size=1
        )

        if "execution" in results:
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
