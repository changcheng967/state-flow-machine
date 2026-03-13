"""
State-Flow Machine (SFM)

A novel post-transformer architecture for code intelligence with 4 specialized systems:
- System 1: Perception (linear-attention decoder)
- System 2: Execution (state slot bank)
- System 3: Structure (graph neural network)
- System 4: Meta (recurrent controller)
"""

from .config import SFMConfig, ExperimentConfig, DEFAULT_CONFIG, SMALL_CONFIG, LARGE_CONFIG
from .model import StateFlowMachine, create_sfm
from .tokenizer import CodeTokenizer

__version__ = "0.1.0"

__all__ = [
    "SFMConfig",
    "ExperimentConfig",
    "DEFAULT_CONFIG",
    "SMALL_CONFIG",
    "LARGE_CONFIG",
    "StateFlowMachine",
    "create_sfm",
    "CodeTokenizer",
]
