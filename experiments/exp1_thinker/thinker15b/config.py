"""config.py — Hyperparameters for Thinker-1.5B v2 from-scratch training.

All dimensions are multiples of 16 for DaVinci Cube (16x16x16 MAC array).
v2 changes: real GRU, slot persistence across layers, standard pre-norm,
512-token slot vocabulary with CE loss, FP32 master weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Thinker15BConfig:
    """Complete configuration for Thinker-1.5B v2 model and training."""

    # ── Vocabulary ──
    vocab_size: int = 32000
    max_seq_len: int = 4096
    turn_token_id: int = 3        # [TURN] separator token (reserved)

    # ── Transformer ──
    hidden_dim: int = 2048
    num_heads: int = 16
    num_layers: int = 24
    intermediate_dim: int = 6144
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # ── SFM Slot Bank ──
    slot_dim: int = 256
    num_slots: int = 16            # 8 variable + 4 control-flow + 2 scratch + 2 global
    slot_num_heads: int = 4
    sfm_layers: Tuple[int, ...] = (5, 11, 17, 23)

    # ── Slot Prediction (v2: CE loss over 512-token slot vocabulary) ──
    slot_vocab_size: int = 512     # discretized slot vocabulary
    slot_pred_weight: float = 0.15

    # ── Training ──
    batch_size: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    time_limit_s: int = 86400
    min_epochs: int = 2

    # ── Convergence ──
    convergence_window: int = 200
    convergence_patience: int = 1000
    convergence_threshold: float = 0.001

    # ── Curriculum (soft mixing by step fraction) ──
    curriculum_p3_start: float = 0.30
    curriculum_p4_start: float = 0.60

    # ── GRPO ──
    grpo_group_size: int = 4
    grpo_weight: float = 0.15

    # ── Multi-turn ──
    max_turns: int = 4             # max turns per training sample

    # ── System ──
    rank_size: int = 4

    def __post_init__(self) -> None:
        """Validate all dimensions are multiples of 16 (DaVinci Cube)."""
        for name, val in [
            ("hidden_dim", self.hidden_dim),
            ("intermediate_dim", self.intermediate_dim),
            ("head_dim", self.head_dim),
            ("slot_dim", self.slot_dim),
        ]:
            if val % 16 != 0:
                raise ValueError(f"{name}={val} is not a multiple of 16")
        assert self.hidden_dim // self.num_heads == self.head_dim, (
            f"hidden_dim/num_heads = {self.hidden_dim // self.num_heads} "
            f"!= head_dim={self.head_dim}"
        )
        assert self.slot_vocab_size % 16 == 0, (
            f"slot_vocab_size={self.slot_vocab_size} is not a multiple of 16"
        )

    @classmethod
    def tiny(cls) -> Thinker15BConfig:
        """CPU verification config: ~50K params."""
        return cls(
            vocab_size=100,
            hidden_dim=128,
            num_heads=2,
            num_layers=2,
            intermediate_dim=384,
            head_dim=64,
            slot_dim=32,
            num_slots=4,
            slot_num_heads=2,
            slot_vocab_size=32,
            sfm_layers=(0, 1),
            max_seq_len=64,
            batch_size=1,
            rank_size=1,
            max_turns=2,
        )

    @classmethod
    def small(cls) -> Thinker15BConfig:
        """Debug config: ~5M params."""
        return cls(
            vocab_size=1000,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            intermediate_dim=1536,
            head_dim=64,
            slot_dim=64,
            num_slots=8,
            slot_num_heads=2,
            slot_vocab_size=64,
            sfm_layers=(1, 3, 5),
            max_seq_len=512,
            batch_size=2,
            max_turns=2,
        )
