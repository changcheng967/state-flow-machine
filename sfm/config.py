"""
SFM Configuration — All hyperparameters in one place.

This module defines all hyperparameters for the State-Flow Machine architecture.
Modify these values to experiment with different model sizes.

ASCEND NPU OPTIMIZATION: All dimensions are multiples of 16 for DaVinci Cube unit.
The Cube unit has a 16x16x16 MAC array, so dimensions aligned to 16 maximize utilization.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFMConfig:
    """Configuration for the full State-Flow Machine model."""

    # Vocabulary and tokenization
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Shared embedding dimension (multiple of 16 for Cube)
    d_model: int = 512

    # Cross-system bridge dimension (multiple of 16)
    d_bridge: int = 256

    # Dropout (applied throughout)
    dropout: float = 0.1

    # === System 1: Perception ===
    perception_num_layers: int = 8
    perception_num_heads: int = 8  # head_dim = 512/8 = 64 (multiple of 16)
    perception_ff_dim: int = 2048  # multiple of 16

    # === System 2: Execution ===
    execution_num_slots: int = 64  # multiple of 16
    execution_slot_dim: int = 128  # multiple of 16
    execution_num_heads: int = 4   # head_dim = 128/4 = 32 (multiple of 16)
    execution_max_ticks: int = 2   # reduced for speed
    execution_halting_threshold: float = 0.5

    # DeltaNet cell parameters (all multiples of 16)
    deltanet_hidden_dim: int = 256
    deltanet_num_heads: int = 4  # head_dim = 256/4 = 64 (multiple of 16)
    deltanet_eigenvalue_init: float = 0.9

    # === System 3: Structure ===
    structure_node_dim: int = 256  # multiple of 16
    structure_edge_dim: int = 128  # multiple of 16
    structure_num_layers: int = 4
    structure_num_heads: int = 4   # head_dim = 256/4 = 64 (multiple of 16)
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
    meta_hidden_dim: int = 256  # multiple of 16
    meta_num_heads: int = 4     # head_dim = 256/4 = 64
    meta_hypothesis_dim: int = 128  # multiple of 16
    meta_plan_stack_depth: int = 8
    meta_verification_threshold: float = 0.8

    # === Training ===
    learning_rate: float = 3e-4  # INCREASED for regression
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    seed: int = 42

    # === Bridge synchronization ===
    bridge_sync_interval: int = 2

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.perception_num_heads == 0, \
            "d_model must be divisible by perception_num_heads"
        assert self.execution_slot_dim % self.execution_num_heads == 0, \
            "execution_slot_dim must be divisible by execution_num_heads"
        assert self.d_model > 0 and self.d_bridge > 0, \
            "Dimensions must be positive"
        # Verify all dimensions are multiples of 16 for Cube optimization
        assert self.d_model % 16 == 0, "d_model must be multiple of 16"
        assert self.d_bridge % 16 == 0, "d_bridge must be multiple of 16"
        assert self.execution_slot_dim % 16 == 0, "execution_slot_dim must be multiple of 16"
        assert self.deltanet_hidden_dim % 16 == 0, "deltanet_hidden_dim must be multiple of 16"

    @classmethod
    def small(cls) -> "SFMConfig":
        """
        Small configuration for testing/debugging.

        ALL dimensions are multiples of 16 for DaVinci Cube optimization.
        """
        return cls(
            vocab_size=1000,
            max_seq_len=512,
            d_model=256,      # 16 * 16
            d_bridge=256,     # 16 * 16 (INCREASED for better capacity)
            dropout=0.1,
            perception_num_layers=2,
            perception_num_heads=4,   # head_dim = 64
            perception_ff_dim=512,    # 16 * 32
            execution_num_slots=64,   # 16 * 4 (INCREASED)
            execution_slot_dim=128,   # 16 * 8
            execution_num_heads=4,    # head_dim = 32
            execution_max_ticks=2,
            deltanet_hidden_dim=256,  # 16 * 16
            deltanet_num_heads=4,     # head_dim = 64
            structure_node_dim=256,   # 16 * 16
            structure_edge_dim=128,   # 16 * 8
            structure_num_layers=2,
            structure_num_heads=4,    # head_dim = 64
            structure_max_nodes=256,
            structure_max_edges=512,
            meta_hidden_dim=256,      # 16 * 16
            meta_num_heads=4,         # head_dim = 64
            meta_hypothesis_dim=128,  # 16 * 8
            meta_plan_stack_depth=4,
            learning_rate=3e-4,
            warmup_steps=500,
        )

    @classmethod
    def base(cls) -> "SFMConfig":
        """Base configuration for production training."""
        return cls()

    @classmethod
    def large(cls) -> "SFMConfig":
        """Large configuration for more capacity (all dimensions multiples of 16)."""
        return cls(
            vocab_size=32000,
            max_seq_len=8192,
            d_model=768,       # 16 * 48
            d_bridge=384,      # 16 * 24
            dropout=0.1,
            perception_num_layers=12,
            perception_num_heads=12,  # head_dim = 64
            perception_ff_dim=3072,   # 16 * 192
            execution_num_slots=128,  # 16 * 8
            execution_slot_dim=192,   # 16 * 12
            execution_num_heads=6,    # head_dim = 32
            execution_max_ticks=8,
            deltanet_hidden_dim=512,  # 16 * 32
            deltanet_num_heads=8,     # head_dim = 64
            structure_node_dim=384,   # 16 * 24
            structure_edge_dim=192,   # 16 * 12
            structure_num_layers=6,
            structure_num_heads=6,    # head_dim = 64
            structure_max_nodes=2048,
            structure_max_edges=8192,
            meta_hidden_dim=384,      # 16 * 24
            meta_num_heads=6,         # head_dim = 64
            meta_hypothesis_dim=192,  # 16 * 12
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
    num_epochs: int = 50  # INCREASED from 10
    eval_every: int = 100
    save_every: int = 500

    # Gradient accumulation
    grad_accum_steps: int = 2  # effective_batch = 32 * 4 * 2 = 256

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
            grad_accum_steps=1,
        )


# Default configs for easy import
DEFAULT_CONFIG = SFMConfig.base()
SMALL_CONFIG = SFMConfig.small()
LARGE_CONFIG = SFMConfig.large()
