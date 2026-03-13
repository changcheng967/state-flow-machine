"""SFM Systems package - the 4 specialized systems."""

from .perception import PerceptionSystem, PerceptionLayer, TokenEmbedding
from .execution import ExecutionSystem
from .structure import StructureSystem, CodeGraph, NODE_TYPES, EDGE_TYPES
from .meta import MetaSystem, HypothesisRegister, PlanStack, VerificationHead

__all__ = [
    # Perception (System 1)
    "PerceptionSystem",
    "PerceptionLayer",
    "TokenEmbedding",
    # Execution (System 2)
    "ExecutionSystem",
    # Structure (System 3)
    "StructureSystem",
    "CodeGraph",
    "NODE_TYPES",
    "EDGE_TYPES",
    # Meta (System 4)
    "MetaSystem",
    "HypothesisRegister",
    "PlanStack",
    "VerificationHead",
]
