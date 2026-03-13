"""
SFM Configuration — All hyperparameters in one place.

This module defines all hyperparameters for the State-Flow Machine architecture.
Modify these values to experiment with different model sizes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFMConfig:
    """Configuration for the full State-Flow Machine model."""

    # Vocabulary and tokenization
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Shared embedding dimension
    d_model: int = 512

    # Cross-system bridge dimension
    d_bridge: int = 256

    # Dropout (applied throughout)
    dropout: float = 0.1

    # === System 1: Perception ===
    perception_num_layers: int = 8
    perception_num_heads: int = 8
    perception_ff_dim: int = 2048

    # === System 2: Execution ===
    execution_num_slots: int = 64  # State Slot Bank size
    execution_slot_dim: int = 128  # Dimension per slot
    execution_num_heads: int = 4
    execution_max_ticks: int = 2  # Max internal ticks per statement (reduced for speed)
    execution_halting_threshold: float = 0.5

    # DeltaNet cell parameters
    deltanet_hidden_dim: int = 256
    deltanet_num_heads: int = 4
    deltanet_eigenvalue_init: float = 0.9  # Initialize eigenvalues close to 1

    # === System 3: Structure ===
    structure_node_dim: int = 256
    structure_edge_dim: int = 128
    structure_num_layers: int = 4
    structure_num_heads: int = 4
    structure_max_nodes: int = 1024
    structure_max_edges: int = 4096

    # Edge types for code graph
    structure_edge_types: tuple = (
        "calls",      # Function calls function
        "imports",    # Module imports module
        "mutates",    # Statement modifies variable
        "reads",      # Statement reads variable
        "defines",    # Scope defines symbol
        "contains",   # Parent contains child (class->method, file->function)
    )

    # === System 4: Meta ===
    meta_hidden_dim: int = 256
    meta_num_heads: int = 4
    meta_hypothesis_dim: int = 128
    meta_plan_stack_depth: int = 8
    meta_verification_threshold: float = 0.8

    # === Training ===
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    seed: int = 42

    # === Bridge synchronization ===
    # Systems exchange info every N perception layers
    bridge_sync_interval: int = 2

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.perception_num_heads == 0, \
            "d_model must be divisible by perception_num_heads"
        assert self.execution_slot_dim % self.execution_num_heads == 0, \
            "execution_slot_dim must be divisible by execution_num_heads"
        assert self.d_model > 0 and self.d_bridge > 0, \
            "Dimensions must be positive"

    @classmethod
    def small(cls) -> "SFMConfig":
        """Small configuration for testing/debugging."""
        return cls(
            vocab_size=1000,
            max_seq_len=512,
            d_model=128,
            d_bridge=64,
            dropout=0.1,
            perception_num_layers=2,
            perception_num_heads=4,
            perception_ff_dim=256,
            execution_num_slots=16,
            execution_slot_dim=64,
            execution_num_heads=2,
            execution_max_ticks=2,
            deltanet_hidden_dim=128,
            deltanet_num_heads=2,
            structure_node_dim=64,
            structure_edge_dim=32,
            structure_num_layers=2,
            structure_num_heads=2,
            structure_max_nodes=256,
            structure_max_edges=512,
            meta_hidden_dim=64,
            meta_num_heads=2,
            meta_hypothesis_dim=32,
            meta_plan_stack_depth=4,
        )

    @classmethod
    def base(cls) -> "SFMConfig":
        """Base configuration for production training."""
        return cls()

    @classmethod
    def large(cls) -> "SFMConfig":
        """Large configuration for more capacity."""
        return cls(
            vocab_size=32000,
            max_seq_len=8192,
            d_model=768,
            d_bridge=384,
            dropout=0.1,
            perception_num_layers=12,
            perception_num_heads=12,
            perception_ff_dim=3072,
            execution_num_slots=128,
            execution_slot_dim=192,
            execution_num_heads=6,
            execution_max_ticks=8,
            deltanet_hidden_dim=512,
            deltanet_num_heads=8,
            structure_node_dim=384,
            structure_edge_dim=192,
            structure_num_layers=6,
            structure_num_heads=6,
            structure_max_nodes=2048,
            structure_max_edges=8192,
            meta_hidden_dim=384,
            meta_num_heads=6,
            meta_hypothesis_dim=192,
            meta_plan_stack_depth=12,
        )


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    # Experiment identification
    name: str = "exp0_state_tracking"
    output_dir: str = "outputs"

    # Data
    train_samples: int = 10000
    val_samples: int = 1000
    max_program_length: int = 20  # Max statements per program
    min_program_length: int = 3

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    eval_every: int = 100
    save_every: int = 500

    # Evaluation
    eval_max_length_multiplier: int = 4  # Test generalization to 4x length

    # Reproducibility
    seed: int = 42

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        """Quick configuration for smoke testing."""
        return cls(
            name="exp0_quick",
            train_samples=100,
            val_samples=20,
            max_program_length=10,
            batch_size=8,
            num_epochs=1,
            eval_every=10,
            save_every=50,
        )


# Default configs for easy import
DEFAULT_CONFIG = SFMConfig.base()
SMALL_CONFIG = SFMConfig.small()
LARGE_CONFIG = SFMConfig.large()
