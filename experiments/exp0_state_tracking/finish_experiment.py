"""
finish_experiment.py - Single-NPU training + full evaluation (v3)

Self-contained: trains all three models (execution + two transformers) on a
single NPU (no DDP), then evaluates all three on length generalization at
1x, 2x, 4x, 8x, 16x, 32x.

v3 changes:
- Fair comparison: transformers use 101-class CE (same as execution model)
- State passing: carry detached recurrent state across training batches
- LR grid search for both execution and transformer models
- Extended evaluation up to 32x (320 ops)

Usage:
    python experiments/exp0_state_tracking/finish_experiment.py
    python experiments/exp0_state_tracking/finish_experiment.py --skip_training
"""

import os
import sys
import json
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])

from sfm.config import SFMConfig
from sfm.systems.execution import ExecutionSystem
from sfm.tokenizer.code_tokenizer import SimpleTokenizer
from sfm.utils.device import get_device, set_seed, synchronize
from baseline_transformer import TransformerEncoderOnly, StateTrackingWrapper
from dataset import create_dataloaders
from evaluate import Evaluator, generate_test_programs
from generate_data import generate_and_save


NUM_CLASSES = 101  # values 0-100


# ---------------------------------------------------------------------------
# Learning rate scheduler (cosine annealing with linear warmup)
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ---------------------------------------------------------------------------
# Training loop for a single transformer on one NPU (UNCHANGED)
# ---------------------------------------------------------------------------

def train_transformer(
    model,
    train_loader,
    val_loader,
    device,
    model_name,
    save_dir,
    num_epochs=50,
    lr=3e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=500,
    use_amp=True,
    num_classes=101,
):
    """Train a transformer model on single NPU, no DDP (v3: 101-class classification)."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # AMP scaler (conservative settings matching train.py)
    if use_amp:
        try:
            scaler = torch.npu.amp.GradScaler(
                init_scale=1024.0, growth_factor=1.5,
                backoff_factor=0.5, growth_interval=500
            )
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(
                init_scale=1024.0, growth_factor=1.5,
                backoff_factor=0.5, growth_interval=500
            )
    else:
        scaler = None

    best_val_acc = 0.0
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print(f"Training {model_name} (v3: classification, {num_classes} classes)")
    print(f"{'=' * 60}")
    print(f"  Parameters: {param_count:,}")
    print(f"  AMP: {use_amp}  Device: {device}")
    print(f"  LR: {lr}  Warmup: {warmup_steps} steps")
    print(f"  Batch size: {train_loader.batch_size}  Steps/epoch: {len(train_loader)}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_raw = batch["final_value"].to(device, non_blocking=True).long()

            mask = (attention_mask == 0)  # True for padding

            # Forward with optional AMP
            if use_amp:
                try:
                    with torch.npu.amp.autocast():
                        logits = model(input_ids, mask=mask)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, mask=mask)
            else:
                logits = model(input_ids, mask=mask)

            # CE loss (classification)
            loss = F.cross_entropy(logits.float(), labels_raw)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            predicted_classes = logits.argmax(dim=-1)
            total_correct += (predicted_classes == labels_raw).sum().item()
            total_samples += labels_raw.size(0)

        synchronize()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        # ---- Validate ----
        model.eval()
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels_raw = batch["final_value"].to(device, non_blocking=True).long()

                mask = (attention_mask == 0)
                predictions = model(input_ids, mask=mask)
                predicted_classes = (predictions * 100).round().long()
                val_correct += (predicted_classes == labels_raw).sum().item()
                val_samples += labels_raw.size(0)

        val_acc = val_correct / val_samples
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"loss={train_loss:.4f}/acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  [{epoch_time:.1f}s]")

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "epoch": epoch,
                "param_count": param_count,
            }, save_path)
            print(f"  -> Saved best (val_acc={val_acc:.4f})")

    print(f"\n  {model_name} done. Best val_acc: {best_val_acc:.4f}")
    return best_val_acc


# ---------------------------------------------------------------------------
# Execution model factory
# ---------------------------------------------------------------------------

def make_execution_model(vocab_size: int, device: torch.device) -> nn.Module:
    """Create execution model with v2 settings."""
    config = SFMConfig.small()
    execution_sys = ExecutionSystem(
        input_dim=config.d_model,          # 256
        hidden_dim=config.deltanet_hidden_dim,  # 256
        num_slots=16,
        slot_dim=256,                      # Increased from 128
        max_ticks=config.execution_max_ticks,
        num_heads=config.execution_num_heads,
        dropout=config.dropout
    )
    model = StateTrackingWrapper(
        execution_sys, vocab_size, num_classes=NUM_CLASSES
    )
    return model


# ---------------------------------------------------------------------------
# Training loop for execution model (v2: classification + aux loss)
# ---------------------------------------------------------------------------

def train_execution(
    model,
    train_loader,
    val_loader,
    device,
    save_dir,
    num_epochs=50,
    lr=1e-3,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=200,
    aux_weight=0.5,
):
    """Train execution model with classification + intermediate state supervision.

    FP32 only (no AMP) — DeltaNet backward through exp() overflows FP16.
    Classification (101 classes) gives sharper gradients than MSE regression.
    Auxiliary loss at every token position provides intermediate state supervision.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print(f"Training Execution Model (State Slots v2)")
    print(f"{'=' * 60}")
    print(f"  Parameters: {param_count:,}")
    print(f"  AMP: False (FP32)  Device: {device}")
    print(f"  LR: {lr}  Warmup: {warmup_steps} steps")
    print(f"  Loss: CE + {aux_weight} * aux_CE  Classes: {NUM_CLASSES}")
    print(f"  Batch size: {train_loader.batch_size}  Steps/epoch: {len(train_loader)}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_final_loss = 0.0
        total_aux_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.time()

        optimizer.zero_grad()
        carry_state = None  # Reset state each epoch (data is shuffled)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels_raw = batch["final_value"].to(device, non_blocking=True).long()
            inter_targets = batch["intermediate_targets"].to(device, non_blocking=True).long()

            # Forward with intermediate predictions and state passing
            final_logits, inter_logits, new_state = model(
                input_ids, attention_mask=attention_mask,
                return_intermediate=True, state=carry_state
            )

            # Carry state forward (detached to prevent cross-batch gradients)
            carry_state = {k: v.detach() for k, v in new_state.items()}

            # Final classification loss: (batch, 101) vs (batch,)
            final_loss = F.cross_entropy(final_logits, labels_raw)

            # Auxiliary intermediate loss at every token position
            # inter_logits: (batch, seq_len, 101), inter_targets: (batch, seq_len)
            aux_loss = F.cross_entropy(
                inter_logits.reshape(-1, NUM_CLASSES),
                inter_targets.reshape(-1),
                ignore_index=-100
            )

            loss = final_loss + aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_final_loss += final_loss.item()
            total_aux_loss += aux_loss.item()
            predicted = final_logits.argmax(dim=-1)
            total_correct += (predicted == labels_raw).sum().item()
            total_samples += labels_raw.size(0)

        synchronize()
        n_batches = len(train_loader)
        train_loss = total_loss / n_batches
        train_acc = total_correct / total_samples

        # ---- Validate ----
        model.eval()
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels_raw = batch["final_value"].to(device, non_blocking=True).long()

                predictions = model(input_ids, attention_mask=attention_mask)
                predicted_classes = (predictions * 100).round().long()
                val_correct += (predicted_classes == labels_raw).sum().item()
                val_samples += labels_raw.size(0)

        val_acc = val_correct / val_samples
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"loss={train_loss:.4f} (f={total_final_loss/n_batches:.3f} "
                  f"a={total_aux_loss/n_batches:.3f})  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"[{epoch_time:.1f}s]")

        # Save best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, "execution_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "epoch": epoch,
                "param_count": param_count,
                "lr": lr,
            }, save_path)
            print(f"  -> Saved best (val_acc={val_acc:.4f})")

    print(f"\n  Execution done. Best val_acc: {best_val_acc:.4f}")
    return best_val_acc


# ---------------------------------------------------------------------------
# Full evaluation of all three models
# ---------------------------------------------------------------------------

def run_full_evaluation(save_dir, device, tokenizer,
                        base_length=10, multipliers=None,
                        samples_per_length=1000, difficulty="easy"):
    """Load all 3 checkpoints, evaluate at multiple lengths, print table."""

    if multipliers is None:
        multipliers = [1, 2, 4, 8, 16, 32]

    config = SFMConfig.small()
    results = {}

    # Model specs: (type, name, is_transformer, constructor kwargs)
    model_specs = [
        ("execution", "State Slots", False, {}),
        ("transformer_fair", "Transformer-Fair", True,
         dict(d_model=128, num_heads=4, num_layers=3, d_ff=256, num_output_classes=NUM_CLASSES)),
        ("transformer_large", "Transformer-Large", True,
         dict(d_model=256, num_heads=8, num_layers=4, d_ff=512, num_output_classes=NUM_CLASSES)),
    ]

    for model_type, model_name, is_transformer, extra_kwargs in model_specs:
        model_path = os.path.join(save_dir, f"{model_type}_best.pt")
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")

        # Build model
        if model_type == "execution":
            execution = ExecutionSystem(
                input_dim=config.d_model,
                hidden_dim=config.deltanet_hidden_dim,
                num_slots=16,
                slot_dim=256,  # Must match training
                max_ticks=config.execution_max_ticks,
                num_heads=config.execution_num_heads,
                dropout=config.dropout
            )
            model = StateTrackingWrapper(execution, tokenizer.vocab_size_actual,
                                         num_classes=NUM_CLASSES)
        else:
            model = TransformerEncoderOnly(
                vocab_size=tokenizer.vocab_size_actual, **extra_kwargs
            )

        # Load checkpoint
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded {model_path} (epoch {ckpt.get('epoch', '?')}, "
                  f"val_acc={ckpt.get('val_accuracy', '?')})")
        else:
            print(f"  WARNING: {model_path} not found, using random weights!")

        model = model.to(device)
        model.eval()
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        evaluator = Evaluator(model, tokenizer, device, model_name,
                              is_transformer=is_transformer)
        results[model_type] = {}

        for mult in multipliers:
            target_length = base_length * mult
            samples = generate_test_programs(
                num_samples=samples_per_length,
                exact_length=target_length,
                seed=42 + mult,
                difficulty=difficulty
            )
            metrics = evaluator.evaluate_samples(samples)
            results[model_type][mult] = metrics
            print(f"  {mult}x ({target_length:3d} ops): "
                  f"EM={metrics['exact_match']:.4f}  "
                  f"Close={metrics['close_match']:.4f}  "
                  f"MSE={metrics['mse']:.6f}  "
                  f"MAE={metrics['mae']:.2f}")

    # ---- Results table ----
    print("\n" + "=" * 100)
    print("GENERALIZATION RESULTS")
    print("=" * 100)
    header = (f"{'Length':<8} {'SS EM':<10} {'TF-F EM':<10} {'TF-L EM':<10} "
              f"{'SS Close':<10} {'TF-F Close':<10} {'TF-L Close':<10}")
    print(header)
    print("-" * len(header))

    for mult in multipliers:
        ss = results["execution"][mult]
        tf = results["transformer_fair"][mult]
        tl = results["transformer_large"][mult]
        print(f"{mult}x{'':<6} {ss['exact_match']:<10.4f} {tf['exact_match']:<10.4f} "
              f"{tl['exact_match']:<10.4f} {ss['close_match']:<10.4f} "
              f"{tf['close_match']:<10.4f} {tl['close_match']:<10.4f}")

    # ---- Generalization ratios ----
    print("\n" + "-" * 60)
    print("GENERALIZATION RATIOS (EM_4x / EM_1x):")
    if 1 in multipliers and 4 in multipliers:
        ss_r = results["execution"][4]["exact_match"] / max(results["execution"][1]["exact_match"], 1e-4)
        tf_r = results["transformer_fair"][4]["exact_match"] / max(results["transformer_fair"][1]["exact_match"], 1e-4)
        tl_r = results["transformer_large"][4]["exact_match"] / max(results["transformer_large"][1]["exact_match"], 1e-4)
        print(f"  State Slots:       {ss_r:.2f}x")
        print(f"  Transformer-Fair:  {tf_r:.2f}x")
        print(f"  Transformer-Large: {tl_r:.2f}x")

        if 8 in multipliers:
            ss_8 = results["execution"][8]["exact_match"] / max(results["execution"][1]["exact_match"], 1e-4)
            tf_8 = results["transformer_fair"][8]["exact_match"] / max(results["transformer_fair"][1]["exact_match"], 1e-4)
            tl_8 = results["transformer_large"][8]["exact_match"] / max(results["transformer_large"][1]["exact_match"], 1e-4)
            print(f"\n  8x ratios:")
            print(f"  State Slots:       {ss_8:.2f}x")
            print(f"  Transformer-Fair:  {tf_8:.2f}x")
            print(f"  Transformer-Large: {tl_8:.2f}x")

        # Verdict at 4x and 8x
        pass_4x = ss_r > tf_r + 0.1
        pass_8x = ss_8 > tf_8 + 0.1 if 8 in multipliers else False
        if pass_4x and pass_8x:
            print(f"\n  VERDICT: PASS - State Slots generalize better at both 4x and 8x")
        elif pass_4x:
            print(f"\n  VERDICT: PARTIAL PASS - Advantage at 4x, not at 8x")
        elif ss_r > tf_r:
            print(f"\n  VERDICT: MARGINAL - Slight advantage at 4x only")
        else:
            print(f"\n  VERDICT: FAIL - No generalization advantage")

    # ---- Save JSON ----
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))
        x_labels = [f"{m}x" for m in multipliers]
        ax.plot(x_labels, [results["execution"][m]["exact_match"] * 100 for m in multipliers],
                'b-o', lw=2, ms=8, label='State Slots')
        ax.plot(x_labels, [results["transformer_fair"][m]["exact_match"] * 100 for m in multipliers],
                'r--s', lw=2, ms=8, label='Transformer-Fair')
        ax.plot(x_labels, [results["transformer_large"][m]["exact_match"] * 100 for m in multipliers],
                'g:^', lw=2, ms=8, label='Transformer-Large')
        ax.set_xlabel('Length Multiplier', fontsize=12)
        ax.set_ylabel('Exact Match Accuracy (%)', fontsize=12)
        ax.set_title('Experiment 0 v3: Fair Classification + Mixed-Length + State Passing', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "length_generalization.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-NPU training + full evaluation (v2)"
    )
    parser.add_argument("--save_dir", type=str, default="outputs/exp0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--difficulty", type=str, default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--skip_lr_search", action="store_true",
                        help="Skip LR grid search, use default lr=1e-3")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    data_dir = os.path.join(args.save_dir, "data")
    train_path = os.path.join(data_dir, "train.json")
    vocab_path = os.path.join(args.save_dir, "tokenizer_vocab.json")

    # ---- Clean stale checkpoints ----
    stale_ckpts = [
        os.path.join(args.save_dir, "execution_best.pt"),
        os.path.join(args.save_dir, "transformer_fair_best.pt"),
        os.path.join(args.save_dir, "transformer_large_best.pt"),
    ]
    existing_stale = [p for p in stale_ckpts if os.path.exists(p)]
    if existing_stale:
        print("Cleaning stale checkpoints...")
        for p in existing_stale:
            os.remove(p)
            print(f"  Deleted {p}")

    # ---- Ensure data is large enough ----
    min_train_samples = 10000
    need_regen = False

    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            existing_train = json.load(f)
        if len(existing_train) < min_train_samples:
            print(f"Existing data has only {len(existing_train)} samples, "
                  f"regenerating with {min_train_samples}...")
            need_regen = True
        else:
            # Validate data format: standard hard programs have at most ~27 lines
            # (5 init + max_program_length ops + 2 query). If programs are longer,
            # the data is stale (e.g. from a mixed-length run) and must be regenerated.
            max_lines = max(s["num_lines"] for s in existing_train[:100])
            if max_lines > 35:
                print(f"Stale data detected (max {max_lines} lines, expected <=27), "
                      f"regenerating with {min_train_samples} standard-length samples...")
                need_regen = True
    else:
        need_regen = True

    if need_regen:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Generating data: {min_train_samples} train / 1000 val, "
              f"difficulty={args.difficulty}, seed=42...")
        generate_and_save(
            output_dir=data_dir,
            train_samples=min_train_samples,
            val_samples=1000,
            max_program_length=20,
            seed=42,
            difficulty=args.difficulty
        )

    # ---- Build / load tokenizer ----
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    corpus = ["\n".join(s["program"]) for s in train_data]
    tokenizer = SimpleTokenizer()
    tokenizer.train(corpus, verbose=True)
    tokenizer.save(vocab_path)
    print(f"Tokenizer: {tokenizer.vocab_size_actual} tokens (saved to {vocab_path})")

    # ---- Create dataloaders ----
    train_loader, val_loader = create_dataloaders(
        data_dir, tokenizer,
        batch_size=args.batch_size,
        max_length=200,
        distributed=False, rank=0, world_size=1
    )
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")

    use_amp = "npu" in str(device)

    # ---- Train all models ----
    if not args.skip_training:
        # ======== LR Grid Search for Execution Model ========
        if not args.skip_lr_search:
            learning_rates = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]
            grid_search_epochs = 10
            subset_size = 2000

            print(f"\n{'=' * 60}")
            print(f"LR Grid Search: {len(learning_rates)} LRs, "
                  f"{grid_search_epochs} epochs each, {subset_size} samples")
            print(f"{'=' * 60}")

            # Create subset dataloader
            actual_subset = min(subset_size, len(train_loader.dataset))
            indices = list(range(actual_subset))
            subset_dataset = Subset(train_loader.dataset, indices)
            subset_loader = DataLoader(
                subset_dataset, batch_size=args.batch_size,
                shuffle=True, drop_last=True
            )

            grid_results = {}
            for lr in learning_rates:
                set_seed(42)
                model = make_execution_model(tokenizer.vocab_size_actual, device).to(device)
                best_val_acc = train_execution(
                    model, subset_loader, val_loader, device,
                    save_dir=args.save_dir,
                    num_epochs=grid_search_epochs,
                    lr=lr, warmup_steps=max(50, 200 // grid_search_epochs * 2),
                )
                grid_results[lr] = best_val_acc
                del model
                if "npu" in str(device):
                    torch.npu.empty_cache()

            print(f"\n{'=' * 60}")
            print("LR Grid Search Results")
            print(f"{'=' * 60}")
            for lr, acc in sorted(grid_results.items()):
                marker = " <-- BEST" if acc == max(grid_results.values()) else ""
                print(f"  lr={lr:<8}  val_acc={acc:.4f}{marker}")

            best_lr = max(grid_results, key=grid_results.get)
            print(f"\n  Selected LR: {best_lr} (val_acc={grid_results[best_lr]:.4f})")
        else:
            best_lr = 1e-3
            print(f"\nSkipping LR grid search, using lr={best_lr}")

        # ======== Train Execution Model (full) ========
        set_seed(42)
        execution_model = make_execution_model(tokenizer.vocab_size_actual, device).to(device)
        print(f"\nExecution model parameters: "
              f"{sum(p.numel() for p in execution_model.parameters()):,}")

        # NPU warmup
        if "npu" in str(device):
            print("\nWarming up NPU (execution model)...")
            execution_model.train()
            warmup_opt = torch.optim.SGD(execution_model.parameters(), lr=0.0)
            warmup_batch = next(iter(train_loader))
            for _ in range(3):
                warmup_opt.zero_grad()
                ids = warmup_batch["input_ids"].to(device, non_blocking=True)
                attn = warmup_batch["attention_mask"].to(device, non_blocking=True)
                out, _, _ = execution_model(ids, attention_mask=attn, return_intermediate=True)
                loss = out.sum()
                loss.backward()
                warmup_opt.step()
            synchronize()
            print("NPU warmup complete (execution)")
            del warmup_opt
            torch.npu.empty_cache()

        train_execution(
            execution_model, train_loader, val_loader, device,
            save_dir=args.save_dir,
            num_epochs=args.epochs, lr=best_lr, warmup_steps=200,
        )
        del execution_model
        if "npu" in str(device):
            torch.npu.empty_cache()

        # ======== Train Transformers (v3: classification) ========
        if "npu" in str(device):
            print("\nWarming up NPU (transformer)...")
            warmup_model = TransformerEncoderOnly(
                tokenizer.vocab_size_actual, d_model=128, num_heads=4,
                num_layers=1, d_ff=256, num_output_classes=NUM_CLASSES
            ).to(device)
            warmup_model.train()
            warmup_opt = torch.optim.SGD(warmup_model.parameters(), lr=0.0)
            warmup_batch = next(iter(train_loader))
            for _ in range(3):
                warmup_opt.zero_grad()
                ids = warmup_batch["input_ids"].to(device, non_blocking=True)
                attn_mask = warmup_batch["attention_mask"].to(device, non_blocking=True)
                mask = (attn_mask == 0)
                if use_amp:
                    try:
                        with torch.npu.amp.autocast():
                            logits = warmup_model(ids, mask=mask)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            logits = warmup_model(ids, mask=mask)
                else:
                    logits = warmup_model(ids, mask=mask)
                loss = F.cross_entropy(logits.float(),
                                       warmup_batch["final_value"].to(device).long())
                loss.backward()
                warmup_opt.step()
            synchronize()
            print("NPU warmup complete")
            del warmup_model, warmup_opt
            torch.npu.empty_cache()

        # ======== LR Grid Search for Transformer-Fair ========
        tf_learning_rates = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]
        tf_grid_epochs = 10
        tf_subset_size = 2000

        print(f"\n{'=' * 60}")
        print(f"Transformer LR Grid Search: {len(tf_learning_rates)} LRs, "
              f"{tf_grid_epochs} epochs each, {tf_subset_size} samples")
        print(f"{'=' * 60}")

        actual_subset = min(tf_subset_size, len(train_loader.dataset))
        indices = list(range(actual_subset))
        tf_subset_dataset = Subset(train_loader.dataset, indices)
        tf_subset_loader = DataLoader(
            tf_subset_dataset, batch_size=args.batch_size,
            shuffle=True, drop_last=True
        )

        tf_grid_results = {}
        for tf_lr in tf_learning_rates:
            set_seed(42)
            tf_model = TransformerEncoderOnly(
                vocab_size=tokenizer.vocab_size_actual,
                d_model=128, num_heads=4, num_layers=3, d_ff=256,
                num_output_classes=NUM_CLASSES
            ).to(device)
            best_va = train_transformer(
                tf_model, tf_subset_loader, val_loader, device,
                "transformer_fair", args.save_dir,
                num_epochs=tf_grid_epochs, lr=tf_lr,
                warmup_steps=max(50, 200 // tf_grid_epochs * 2),
                use_amp=use_amp, num_classes=NUM_CLASSES,
            )
            tf_grid_results[tf_lr] = best_va
            del tf_model
            if "npu" in str(device):
                torch.npu.empty_cache()

        print(f"\n{'=' * 60}")
        print("Transformer LR Grid Search Results")
        print(f"{'=' * 60}")
        for tf_lr, acc in sorted(tf_grid_results.items()):
            marker = " <-- BEST" if acc == max(tf_grid_results.values()) else ""
            print(f"  lr={tf_lr:<8}  val_acc={acc:.4f}{marker}")

        best_tf_lr = max(tf_grid_results, key=tf_grid_results.get)
        print(f"\n  Selected LR: {best_tf_lr} (val_acc={tf_grid_results[best_tf_lr]:.4f})")

        # ======== Train Transformer-Fair (full) ========
        set_seed(42)
        tf_fair = TransformerEncoderOnly(
            vocab_size=tokenizer.vocab_size_actual,
            d_model=128, num_heads=4, num_layers=3, d_ff=256,
            num_output_classes=NUM_CLASSES
        ).to(device)

        train_transformer(
            tf_fair, train_loader, val_loader, device,
            "transformer_fair", args.save_dir,
            num_epochs=args.epochs, lr=best_tf_lr,
            warmup_steps=args.warmup_steps, use_amp=use_amp,
            num_classes=NUM_CLASSES,
        )
        del tf_fair
        if "npu" in str(device):
            torch.npu.empty_cache()

        # ======== Train Transformer-Large ========
        set_seed(42)
        tf_large = TransformerEncoderOnly(
            vocab_size=tokenizer.vocab_size_actual,
            d_model=256, num_heads=8, num_layers=4, d_ff=512,
            num_output_classes=NUM_CLASSES
        ).to(device)

        train_transformer(
            tf_large, train_loader, val_loader, device,
            "transformer_large", args.save_dir,
            num_epochs=args.epochs, lr=args.lr,
            warmup_steps=args.warmup_steps, use_amp=use_amp,
            num_classes=NUM_CLASSES,
        )
        del tf_large

    # ---- Full evaluation ----
    print("\n" + "=" * 60)
    print("FULL EVALUATION (all 3 models)")
    print("=" * 60)

    run_full_evaluation(
        args.save_dir, device, tokenizer,
        base_length=10,
        multipliers=[1, 2, 4, 8, 16, 32],
        samples_per_length=args.eval_samples,
        difficulty=args.difficulty
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
