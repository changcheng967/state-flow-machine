"""
Dataset for State Tracking Experiment

PyTorch Dataset for loading and processing state tracking programs.

FIXED TENSOR SHAPES: All samples padded to EXACTLY max_length for NPU graph caching.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Dict, Optional, Tuple
import json
import os
import sys

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])
from sfm.tokenizer.code_tokenizer import SimpleTokenizer


class StateTrackingDataset(Dataset):
    """
    Dataset for state tracking experiment.

    Each sample is a program with a target variable and its final value.
    ALL samples are padded to EXACTLY max_length for consistent tensor shapes.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 256
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file.
            tokenizer: Tokenizer (SimpleTokenizer recommended for speed).
            max_length: EXACT sequence length for all samples.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with FIXED shape.

        Args:
            idx: Sample index.

        Returns:
            Dict with tensors of fixed shape (max_length,).
        """
        sample = self.data[idx]

        # Format program as input
        program_lines = sample["program"]
        final_value = sample["final_value"]

        # Tokenize each line separately to track per-line boundaries
        line_token_ids = []
        for line in program_lines:
            line_token_ids.append(self.tokenizer.encode(line))

        # Flatten: concatenate all line tokens
        all_token_ids = []
        for line_ids in line_token_ids:
            all_token_ids.extend(line_ids)

        # TRUNCATE to max_length
        all_token_ids = all_token_ids[:self.max_length]

        # Get padding token
        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)

        # Create attention mask BEFORE padding (1 for real tokens)
        attention_mask = [1] * len(all_token_ids)

        # PAD to EXACTLY max_length
        padding_length = self.max_length - len(all_token_ids)
        all_token_ids = all_token_ids + [pad_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        # Build intermediate_targets: per-token target variable value
        # -100 = ignore index for cross_entropy (padding and first-tokens)
        intermediate_states = sample.get("intermediate_states", [])
        intermediate_targets = torch.full((self.max_length,), -100, dtype=torch.long)

        if len(intermediate_states) > 0 and len(line_token_ids) > 0:
            # Map each token to its line, assign that line's intermediate state
            # Comment lines (starting with #) have no trace entry, use last state
            trace_idx = 0
            last_state_value = intermediate_states[-1] if intermediate_states else 0
            pos = 0
            for line_idx, line_ids in enumerate(line_token_ids):
                # Determine the state value for this line
                if not program_lines[line_idx].strip().startswith("#"):
                    # Non-comment line: use corresponding trace entry
                    if trace_idx < len(intermediate_states):
                        state_val = intermediate_states[trace_idx]
                        trace_idx += 1
                    else:
                        state_val = last_state_value
                    last_state_value = state_val
                else:
                    # Comment line: state doesn't change
                    state_val = last_state_value

                # Assign state value to all tokens in this line
                for _ in line_ids:
                    if pos < self.max_length:
                        intermediate_targets[pos] = max(0, min(100, state_val))
                        pos += 1

        # Clamp final value
        final_value = max(0, min(100, final_value))
        final_value_normalized = final_value / 100.0

        return {
            "input_ids": torch.tensor(all_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "final_value": torch.tensor(final_value, dtype=torch.long),
            "final_value_normalized": torch.tensor(final_value_normalized, dtype=torch.float),
            "intermediate_targets": intermediate_targets,
        }


def create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing train.json and val.json.
        tokenizer: Tokenizer.
        batch_size: Batch size per GPU.
        max_length: EXACT sequence length for all samples.
        num_workers: Number of data loading workers.
        distributed: Whether to use DistributedSampler.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")

    train_dataset = StateTrackingDataset(
        train_path,
        tokenizer,
        max_length=max_length
    )

    val_dataset = StateTrackingDataset(
        val_path,
        tokenizer,
        max_length=max_length
    )

    # Create samplers
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # pin_memory=False for NPU (only works for CUDA)
    pin_memory = False

    # Ascend optimization: num_workers=4 with persistent_workers
    effective_num_workers = num_workers if num_workers > 0 else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if not distributed else False,
        sampler=train_sampler if distributed else None,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Ensure consistent batch sizes
        persistent_workers=True if effective_num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler if distributed else None,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if effective_num_workers > 0 else False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("=" * 60)
    print("State Tracking Dataset Test")
    print("=" * 60)

    from generate_data import SimpleProgramGenerator
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate test data
        generator = SimpleProgramGenerator(num_variables=5, max_program_length=10, seed=42, difficulty="easy")
        train_data = generator.generate_dataset(100)
        val_data = generator.generate_dataset(20)

        train_path = os.path.join(tmpdir, "train.json")
        val_path = os.path.join(tmpdir, "val.json")

        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(val_path, 'w') as f:
            json.dump(val_data, f)

        print(f"\n1. Generated test data")

        # Create SimpleTokenizer
        print("\n2. Creating SimpleTokenizer...")
        tokenizer = SimpleTokenizer()
        corpus = ["\n".join(s["program"]) for s in train_data + val_data]
        tokenizer.train(corpus, verbose=True)
        print(f"   Vocabulary size: {tokenizer.vocab_size_actual}")

        # Test dataset
        print("\n3. Testing StateTrackingDataset...")
        dataset = StateTrackingDataset(train_path, tokenizer, max_length=64)
        print(f"   Dataset size: {len(dataset)}")

        # Check all samples have same shape
        shapes = set()
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            shapes.add(tuple(sample['input_ids'].shape))
        print(f"   Unique shapes (should be 1): {len(shapes)} - {shapes}")

        # Test dataloaders
        print("\n4. Testing dataloaders...")
        train_loader, val_loader = create_dataloaders(
            tmpdir,
            tokenizer,
            batch_size=8,
            max_length=64
        )

        # Check all batches have same shape
        batch_shapes = set()
        for batch in train_loader:
            batch_shapes.add(tuple(batch['input_ids'].shape))
        print(f"   Unique batch shapes (should be 1): {len(batch_shapes)} - {batch_shapes}")

        batch = next(iter(train_loader))
        print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"   Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"   Batch final_value shape: {batch['final_value'].shape}")

        print("\n" + "=" * 60)
        print("All Dataset tests passed!")
        print("=" * 60)
