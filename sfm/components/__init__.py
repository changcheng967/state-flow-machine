"""SFM Components package - reusable building blocks."""

from .deltanet_cell import DeltaNetCell, DeltaNetLayer, DeltaNetStack
from .state_slots import StateSlotBank, StateSlotLayer
from .linear_attention import (
    LinearAttention,
    CausalLinearAttention,
    LinearAttentionBlock,
    FeatureMap
)
from .graph_attention import (
    GraphAttentionLayer,
    GraphAttentionNetwork,
    DynamicGraphUpdater,
    CodeGraphNodeEncoder,
    EdgeTypeEmbedding
)
from .adaptive_halting import (
    AdaptiveHalting,
    AdaptiveProcessor,
    StepController,
    BudgetAwareHalting
)
from .cross_system_bridge import (
    SystemBridge,
    CrossSystemBridge,
    BridgeSynchronizer,
    BridgedSystemOutput
)

__all__ = [
    # DeltaNet
    "DeltaNetCell",
    "DeltaNetLayer",
    "DeltaNetStack",
    # State Slots
    "StateSlotBank",
    "StateSlotLayer",
    # Linear Attention
    "LinearAttention",
    "CausalLinearAttention",
    "LinearAttentionBlock",
    "FeatureMap",
    # Graph Attention
    "GraphAttentionLayer",
    "GraphAttentionNetwork",
    "DynamicGraphUpdater",
    "CodeGraphNodeEncoder",
    "EdgeTypeEmbedding",
    # Adaptive Halting
    "AdaptiveHalting",
    "AdaptiveProcessor",
    "StepController",
    "BudgetAwareHalting",
    # Cross-System Bridge
    "SystemBridge",
    "CrossSystemBridge",
    "BridgeSynchronizer",
    "BridgedSystemOutput",
]
