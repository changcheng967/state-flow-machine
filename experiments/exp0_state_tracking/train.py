"""
Training Script for State Tracking Experiment

Trains both SFM Execution System (State Slots) and Transformer baseline
for comparison on the state tracking task.

FEATURES:
- Multi-NPU DDP support (HCCL backend for Ascend)
- Automatic Mixed Precision (AMP) for FP16 training
- Gradient accumulation for larger effective batch sizes
- Proper NPU warmup with real batch shapes
- Fixed tensor shapes to avoid graph recompilation
- Detailed timing for all phases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from sfm.tokenizer.code_tokenizer import SimpleTokenizer
from baseline_transformer import TransformerEncoderOnly
from dataset import create_dataloaders
from generate_data import generate_and_save


class StateTrackingWrapper(nn.Module):
    """Wraps ExecutionSystem for state tracking classification."""

    def __init__(
        self,
        execution_system: ExecutionSystem,
        vocab_size: int,
        num_classes: int = 500
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
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Trainer:
    """Trainer with AMP, gradient accumulation, and detailed timing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        model_name: str = "model",
        is_transformer: bool = False,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        rank: int = 0
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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(train_loader) * 10, eta_min=learning_rate * 0.01)
        self.criterion = nn.CrossEntropyLoss()

        # AMP scaler
        if use_amp:
            try:
                self.scaler = torch.npu.amp.GradScaler()
            except AttributeError:
                # Fallback for older PyTorch
                self.scaler = torch.cuda.amp.GradScaler()
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
        """Train for one epoch with gradient accumulation and AMP."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        epoch_start = time.time()
        if self.rank == 0:
            print(f"  Training started at {time.strftime('%H:%M:%S')}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["final_value"].to(self.device)

            mask = self._get_mask(attention_mask)

            # Forward with AMP
            logits = self._forward_with_amp(input_ids, mask)

            # Clamp labels
            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            # Compute loss
            if self.use_amp:
                try:
                    with torch.npu.amp.autocast():
                        loss = self.criterion(logits, labels)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)

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
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            self.global_step += 1

            # Progress indicator
            if self.rank == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
                if batch_idx == 0:
                    batch_time = time.time() - epoch_start
                    print(f' [first batch: {batch_time:.2f}s]')

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        if self.rank == 0:
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
            labels = batch["final_value"].to(self.device)

            mask = self._get_mask(attention_mask)

            if self.is_transformer:
                logits = self.model(input_ids, mask=mask)
            else:
                logits = self.model(input_ids, attention_mask=mask)

            num_classes = logits.size(-1)
            labels = labels.clamp(0, num_classes - 1)

            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

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
                val_metrics = self.evaluate(self.val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Accuracy: {val_metrics['accuracy']:.4f}")

                # Save best model
                if save_dir and val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    # Get underlying model from DDP wrapper
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
    """Initialize distributed training with HCCL backend, fallback to gloo."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Try HCCL first (Ascend), fallback to gloo
    backend = 'hccl'
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Initialized distributed training with backend: {backend}")
    except Exception as e:
        print(f"[Rank {rank}] HCCL backend failed ({e}), falling back to gloo")
        backend = 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Initialized distributed training with backend: {backend}")

    torch.npu.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def warmup_npu(model, train_loader, device, is_transformer: bool = False):
    """Warmup NPU with real batch to compile graphs."""
    print("Warming up NPU with real batch shape...")
    warmup_batch = next(iter(train_loader))
    input_ids = warmup_batch["input_ids"].to(device)
    attention_mask = warmup_batch["attention_mask"].to(device)
    labels = warmup_batch["final_value"].to(device).clamp(0, 499)

    model.train()

    # Forward
    if is_transformer:
        mask = (attention_mask == 0)
        logits = model(input_ids, mask=mask)
    else:
        logits = model(input_ids, attention_mask=attention_mask)

    # Backward
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    model.zero_grad(set_to_none=True)

    # Synchronize to ensure compilation finishes
    try:
        torch.npu.synchronize()
    except AttributeError:
        pass

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
    """
    Run the full state tracking experiment.
    """
    timings = {}

    # Rank 0 creates directories and generates data
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Generate data
        print("Generating data...")
        t0 = time.time()
        generate_and_save(
            output_dir=data_dir,
            train_samples=config.train_samples,
            val_samples=config.val_samples,
            max_program_length=config.max_program_length,
            seed=config.seed,
            difficulty="easy"
        )
        timings["data_generation"] = time.time() - t0
    else:
        data_dir = os.path.join(save_dir, "data")

    # Synchronize in distributed mode
    if distributed:
        dist.barrier()

    # Create tokenizer (SimpleTokenizer for speed)
    if rank == 0:
        print("\nCreating tokenizer...")
    t0 = time.time()
    with open(os.path.join(data_dir, "train.json"), 'r') as f:
        train_data = json.load(f)
    corpus = ["\n".join(s["program"]) for s in train_data]
    tokenizer = SimpleTokenizer()
    tokenizer.train(corpus, verbose=(rank == 0))
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
    grad_accum_steps = 4 if distributed else 1
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
        num_classes=500
    ).to(device)

    if distributed:
        execution_wrapper = DDP(execution_wrapper, device_ids=[rank])

    if rank == 0:
        # Get param count (unwrap DDP if needed)
        model_for_count = execution_wrapper.module if hasattr(execution_wrapper, 'module') else execution_wrapper
        print(f"Execution System parameters: {model_for_count.count_parameters():,}")

    # NPU warmup
    if "npu" in str(device):
        model_to_warmup = execution_wrapper.module if hasattr(execution_wrapper, 'module') else execution_wrapper
        warmup_npu(model_to_warmup, train_loader, device, is_transformer=False)
    timings["model_creation_and_warmup"] = time.time() - t0

    execution_trainer = Trainer(
        execution_wrapper,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        model_name="execution",
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        rank=rank
    )

    t0 = time.time()
    execution_history = execution_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if rank == 0 else None
    )
    timings["execution_training"] = time.time() - t0
    results["execution"] = execution_history

    # ========== Train Transformer Baseline ==========
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Transformer Baseline")
        print("=" * 60)

    t0 = time.time()
    transformer = TransformerEncoderOnly(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=sfm_config.d_model,
        num_heads=4,
        num_layers=4,
        d_ff=sfm_config.d_model * 4,
        num_output_classes=500
    ).to(device)

    if distributed:
        transformer = DDP(transformer, device_ids=[rank])

    if rank == 0:
        model_for_count = transformer.module if hasattr(transformer, 'module') else transformer
        print(f"Transformer parameters: {model_for_count.count_parameters():,}")

    # NPU warmup
    if "npu" in str(device):
        model_to_warmup = transformer.module if hasattr(transformer, 'module') else transformer
        warmup_npu(model_to_warmup, train_loader, device, is_transformer=True)
    timings["transformer_creation_and_warmup"] = time.time() - t0

    transformer_trainer = Trainer(
        transformer,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        model_name="transformer",
        is_transformer=True,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        rank=rank
    )

    t0 = time.time()
    transformer_history = transformer_trainer.train(
        num_epochs=config.num_epochs,
        save_dir=save_dir if rank == 0 else None
    )
    timings["transformer_training"] = time.time() - t0
    results["transformer"] = transformer_history

    # Save results (rank 0 only)
    if rank == 0:
        results["timings"] = timings
        results_path = os.path.join(save_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Print timing breakdown
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
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--save_dir", type=str, default="outputs/exp0", help="Save directory")
    # NOTE: Do NOT add --local-rank here. torchrun sets LOCAL_RANK via environment variable.
    args = parser.parse_args()

    # Get local_rank from environment variable (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        # Distributed training (launched via torchrun)
        setup_distributed(local_rank, world_size)
        device = torch.device(f"npu:{local_rank}")

        if args.quick:
            exp_config = ExperimentConfig.quick()
            sfm_config = SFMConfig.small()
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
        # Single NPU
        set_seed(42)
        device = get_device()

        if args.quick:
            exp_config = ExperimentConfig.quick()
            sfm_config = SFMConfig.small()
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

        results = run_experiment(
            config=exp_config,
            sfm_config=sfm_config,
            device=device,
            save_dir=args.save_dir,
            distributed=False,
            rank=0,
            world_size=1
        )

        # Print summary
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
