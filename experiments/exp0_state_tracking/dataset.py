"""
Dataset for State Tracking Experiment

PyTorch Dataset for loading and processing state tracking programs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import json
import os
import sys

sys.path.insert(0, str(__file__).rsplit('experiments', 1)[0])
from sfm.tokenizer.code_tokenizer import CodeTokenizer


class StateTrackingDataset(Dataset):
    """
    Dataset for state tracking experiment.

    Each sample is a program with a target variable and its final value.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_length: int = 256,
        include_trace: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file.
            tokenizer: Code tokenizer.
            max_length: Maximum sequence length.
            include_trace: Whether to include execution traces.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_trace = include_trace

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with input_ids, labels, and metadata.
        """
        sample = self.data[idx]

        # Format program as input
        program_lines = sample["program"]
        target_var = sample["target_variable"]
        final_value = sample["final_value"]

        # Create input text
        input_text = "\n".join(program_lines)

        # Tokenize
        input_ids = self.tokenizer.encode(input_text)
        input_ids = input_ids[:self.max_length - 1]

        # Create labels (predict final value)
        # The label is the final value tokenized
        label_text = str(final_value)
        label_ids = self.tokenizer.encode(label_text)

        # Pad input
        pad_id = self.tokenizer.special_tokens["<pad>"]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(pad_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids[0] if label_ids else 0, dtype=torch.long),
            "final_value": torch.tensor(final_value, dtype=torch.long),
            "target_variable": target_var,
            "num_lines": sample["num_lines"]
        }


class StateTrackingDatasetWithTrace(Dataset):
    """
    Dataset that includes execution traces for training state slots.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_length: int = 256,
        num_variables: int = 5
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file.
            tokenizer: Code tokenizer.
            max_length: Maximum sequence length.
            num_variables: Number of variables to track.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_variables = num_variables

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with execution trace.
        """
        sample = self.data[idx]
        program_lines = sample["program"]
        trace = sample.get("trace", [])
        final_value = sample["final_value"]

        # Create state tracking targets
        # For each step, we want to predict the state after that step
        state_targets = []

        for step in trace:
            # Create state vector (values of each variable)
            state = step.get("state", {})
            state_vec = []
            for i in range(self.num_variables):
                var = chr(ord('a') + i)
                state_vec.append(state.get(var, 0))
            state_targets.append(state_vec)

        # Pad state targets
        while len(state_targets) < self.max_length:
            state_targets.append([0] * self.num_variables)

        state_targets = state_targets[:self.max_length]

        # Tokenize input
        input_text = "\n".join(program_lines)
        input_ids = self.tokenizer.encode(input_text)
        input_ids = input_ids[:self.max_length]

        # Pad input
        pad_id = self.tokenizer.special_tokens["<pad>"]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(pad_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(final_value, dtype=torch.long),
            "state_targets": torch.tensor(state_targets, dtype=torch.float),
            "num_lines": sample["num_lines"]
        }


def create_dataloaders(
    data_dir: str,
    tokenizer: CodeTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    include_trace: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing train.json and val.json.
        tokenizer: Code tokenizer.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        include_trace: Whether to include execution traces.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")

    DatasetClass = StateTrackingDatasetWithTrace if include_trace else StateTrackingDataset

    train_dataset = DatasetClass(
        train_path,
        tokenizer,
        max_length=max_length
    )

    val_dataset = DatasetClass(
        val_path,
        tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("=" * 60)
    print("State Tracking Dataset Test")
    print("=" * 60)

    # First generate some test data
    from generate_data import SimpleProgramGenerator

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate test data
        generator = SimpleProgramGenerator(num_variables=5, max_program_length=10, seed=42)
        train_data = generator.generate_dataset(100, include_trace=True)
        val_data = generator.generate_dataset(20, include_trace=True)

        train_path = os.path.join(tmpdir, "train.json")
        val_path = os.path.join(tmpdir, "val.json")

        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(val_path, 'w') as f:
            json.dump(val_data, f)

        print(f"\n1. Generated test data in {tmpdir}")

        # Create tokenizer
        print("\n2. Creating tokenizer...")
        tokenizer = CodeTokenizer(vocab_size=1000, min_freq=1)
        corpus = ["\n".join(s["program"]) for s in train_data + val_data]
        tokenizer.train(corpus, verbose=False)
        print(f"   Vocabulary size: {tokenizer.vocab_size_actual}")

        # Test basic dataset
        print("\n3. Testing StateTrackingDataset...")
        dataset = StateTrackingDataset(train_path, tokenizer, max_length=64)
        print(f"   Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"   Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"   Sample labels: {sample['labels'].item()}")
        print(f"   Sample final_value: {sample['final_value'].item()}")

        # Test dataset with trace
        print("\n4. Testing StateTrackingDatasetWithTrace...")
        dataset_trace = StateTrackingDatasetWithTrace(train_path, tokenizer, max_length=64)
        sample = dataset_trace[0]
        print(f"   Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"   Sample state_targets shape: {sample['state_targets'].shape}")

        # Test dataloaders
        print("\n5. Testing dataloaders...")
        train_loader, val_loader = create_dataloaders(
            tmpdir,
            tokenizer,
            batch_size=8,
            max_length=64,
            include_trace=True
        )

        batch = next(iter(train_loader))
        print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"   Batch state_targets shape: {batch['state_targets'].shape}")
        print(f"   Batch labels shape: {batch['labels'].shape}")

        print("\n" + "=" * 60)
        print("All Dataset tests passed!")
        print("=" * 60)
