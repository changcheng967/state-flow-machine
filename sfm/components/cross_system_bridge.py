"""
Cross-System Bridge

Enables communication between the 4 systems (Perception, Execution, Structure, Meta).
Every 2 perception layers, all systems exchange information via projection
to a shared 256d space.

This modularity allows systems to be tested independently while still
enabling emergent behaviors from system interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any


class SystemBridge(nn.Module):
    """
    Single-direction bridge from one system to another.

    Projects from source system's dimension to bridge dimension,
    then to target system's dimension.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        bridge_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize bridge.

        Args:
            source_dim: Source system dimension.
            target_dim: Target system dimension.
            bridge_dim: Shared bridge dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.bridge_dim = bridge_dim

        # Project to bridge dimension
        self.to_bridge = nn.Sequential(
            nn.Linear(source_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Project from bridge dimension
        self.from_bridge = nn.Sequential(
            nn.Linear(bridge_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.Dropout(dropout)
        )

    def to_bridge_space(self, x: torch.Tensor) -> torch.Tensor:
        """Project source to bridge dimension."""
        return self.to_bridge(x)

    def from_bridge_space(self, x: torch.Tensor) -> torch.Tensor:
        """Project from bridge to target dimension."""
        return self.from_bridge(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full bridge transformation.

        Args:
            x: Source tensor (batch, ..., source_dim).

        Returns:
            Target tensor (batch, ..., target_dim).
        """
        bridged = self.to_bridge(x)
        return self.from_bridge(bridged)


class CrossSystemBridge(nn.Module):
    """
    Full cross-system bridge connecting all 4 systems.

    Systems:
    - Perception (System 1): Linear attention decoder
    - Execution (System 2): State Slot Bank
    - Structure (System 3): Graph attention
    - Meta (System 4): Recurrent controller

    Each system can send and receive from every other system,
    with all communication going through the shared bridge dimension.
    """

    def __init__(
        self,
        perception_dim: int,
        execution_dim: int,
        structure_dim: int,
        meta_dim: int,
        bridge_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize cross-system bridge.

        Args:
            perception_dim: System 1 dimension.
            execution_dim: System 2 dimension.
            structure_dim: System 3 dimension.
            meta_dim: System 4 dimension.
            bridge_dim: Shared bridge dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.perception_dim = perception_dim
        self.execution_dim = execution_dim
        self.structure_dim = structure_dim
        self.meta_dim = meta_dim
        self.bridge_dim = bridge_dim

        # System dimensions mapping
        self.system_dims = {
            "perception": perception_dim,
            "execution": execution_dim,
            "structure": structure_dim,
            "meta": meta_dim
        }

        # Bridges from each system to bridge space
        self.to_bridge = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, bridge_dim),
                nn.LayerNorm(bridge_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for name, dim in self.system_dims.items()
        })

        # Bridges from bridge space to each system
        self.from_bridge = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(bridge_dim, dim),
                nn.LayerNorm(dim),
                nn.Dropout(dropout)
            )
            for name, dim in self.system_dims.items()
        })

        # Attention for combining bridge signals
        self.bridge_attention = nn.MultiheadAttention(
            embed_dim=bridge_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Gating for residual connections
        self.gates = nn.ModuleDict({
            name: nn.Linear(dim + dim, dim)
            for name, dim in self.system_dims.items()
        })

    def project_to_bridge(
        self,
        system_name: str,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project system output to bridge space.

        Args:
            system_name: Name of the source system.
            x: System output tensor.

        Returns:
            Tensor in bridge dimension.
        """
        return self.to_bridge[system_name](x)

    def project_from_bridge(
        self,
        system_name: str,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project from bridge space to system dimension.

        Args:
            system_name: Name of the target system.
            x: Tensor in bridge dimension.

        Returns:
            Tensor in target system dimension.
        """
        return self.from_bridge[system_name](x)

    def compute_bridge_attention(
        self,
        bridge_signals: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention-weighted combination of bridge signals.

        Args:
            bridge_signals: Dict mapping system names to bridge tensors.
                           Each tensor is (batch, seq_len, bridge_dim) or (batch, bridge_dim).

        Returns:
            Combined bridge tensor.
        """
        # Stack signals
        system_names = list(bridge_signals.keys())
        signals = torch.stack([bridge_signals[name] for name in system_names], dim=1)
        # If inputs are (batch, seq_len, dim), signals is (batch, num_systems, seq_len, dim)
        # If inputs are (batch, dim), signals is (batch, num_systems, dim)

        if signals.dim() == 4:
            # Flatten: (batch, num_systems, seq_len, dim) -> (batch, num_systems * seq_len, dim)
            batch_size, num_systems, seq_len, dim = signals.shape
            signals_flat = signals.view(batch_size, num_systems * seq_len, dim)

            # Self-attention
            combined_flat, _ = self.bridge_attention(signals_flat, signals_flat, signals_flat)

            # Unflatten: (batch, num_systems * seq_len, dim) -> (batch, num_systems, seq_len, dim)
            combined = combined_flat.view(batch_size, num_systems, seq_len, dim)
        else:
            # 3D: (batch, num_systems, dim)
            combined, _ = self.bridge_attention(signals, signals, signals)

        return combined, system_names

    def apply_gate(
        self,
        system_name: str,
        original: torch.Tensor,
        update: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learned gate for residual connection.

        Args:
            system_name: Name of the system.
            original: Original system state.
            update: Update from bridge.

        Returns:
            Gated combination.
        """
        gate_input = torch.cat([original, update], dim=-1)
        gate = torch.sigmoid(self.gates[system_name](gate_input))
        return gate * original + (1 - gate) * update

    def forward(
        self,
        perception_state: torch.Tensor,
        execution_state: torch.Tensor,
        structure_state: torch.Tensor,
        meta_state: torch.Tensor,
        return_bridge_signals: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Cross-system communication.

        Args:
            perception_state: System 1 state (batch, seq, perception_dim).
            execution_state: System 2 state (batch, seq, execution_dim).
            structure_state: System 3 state (batch, seq, structure_dim).
            meta_state: System 4 state (batch, seq, meta_dim).
            return_bridge_signals: Whether to return bridge signals.

        Returns:
            Tuple of updated states for all 4 systems.
        """
        states = {
            "perception": perception_state,
            "execution": execution_state,
            "structure": structure_state,
            "meta": meta_state
        }

        # Project all to bridge space
        bridge_signals = {
            name: self.project_to_bridge(name, state)
            for name, state in states.items()
        }

        # Compute attention-weighted combination
        combined, system_names = self.compute_bridge_attention(bridge_signals)

        # Map back to system-specific signals
        system_to_idx = {name: i for i, name in enumerate(system_names)}

        # Update each system with gated combination
        updated_states = {}
        for name, original_state in states.items():
            idx = system_to_idx[name]

            # Handle both 3D and 4D combined tensor
            if combined.dim() == 4:
                # (batch, num_systems, seq_len, dim)
                bridge_update = combined[:, idx, :, :]  # (batch, seq_len, bridge_dim)
            else:
                # (batch, num_systems, dim)
                bridge_update = combined[:, idx, :]  # (batch, bridge_dim)

            # Project to system dimension
            update = self.project_from_bridge(name, bridge_update)

            # Handle dimension mismatch for residual
            if update.dim() == 2 and original_state.dim() == 3:
                # Expand to sequence dimension
                update = update.unsqueeze(1).expand(-1, original_state.size(1), -1)

            # Apply gating
            updated_states[name] = self.apply_gate(name, original_state, update)

        result = (
            updated_states["perception"],
            updated_states["execution"],
            updated_states["structure"],
            updated_states["meta"]
        )

        if return_bridge_signals:
            return result + (bridge_signals,)

        return result + (None,)


class BridgeSynchronizer:
    """
    Manages synchronization timing for bridge updates.

    Systems exchange info every N perception layers.
    """

    def __init__(
        self,
        sync_interval: int = 2,
        total_perception_layers: int = 8
    ):
        """
        Initialize synchronizer.

        Args:
            sync_interval: Number of perception layers between syncs.
            total_perception_layers: Total number of perception layers.
        """
        self.sync_interval = sync_interval
        self.total_layers = total_perception_layers
        self.sync_points = list(range(sync_interval - 1, total_perception_layers, sync_interval))

    def should_sync(self, layer_idx: int) -> bool:
        """
        Check if systems should sync at this layer.

        Args:
            layer_idx: Current perception layer index (0-indexed).

        Returns:
            True if sync should happen.
        """
        return layer_idx in self.sync_points

    def get_next_sync_point(self, layer_idx: int) -> Optional[int]:
        """
        Get the next sync point after current layer.

        Args:
            layer_idx: Current layer index.

        Returns:
            Next sync point or None if no more.
        """
        for point in self.sync_points:
            if point > layer_idx:
                return point
        return None


class BridgedSystemOutput(nn.Module):
    """
    Wrapper that handles output projection for bridged systems.
    """

    def __init__(
        self,
        system_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize output wrapper.

        Args:
            system_dim: System's internal dimension.
            output_dim: Final output dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.output_proj = nn.Sequential(
            nn.LayerNorm(system_dim),
            nn.Linear(system_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project system output.

        Args:
            x: System state (batch, seq, system_dim).

        Returns:
            Output tensor (batch, seq, output_dim).
        """
        return self.output_proj(x)


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Cross-System Bridge Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    perception_dim = 256
    execution_dim = 128
    structure_dim = 128
    meta_dim = 128
    bridge_dim = 64  # Reduced for testing

    # Test SystemBridge
    print("\n1. Testing SystemBridge...")
    sys_bridge = SystemBridge(perception_dim, execution_dim, bridge_dim)

    source = torch.randn(batch_size, seq_len, perception_dim)
    target = sys_bridge(source)

    print(f"   Source shape: {source.shape}")
    print(f"   Target shape: {target.shape}")
    assert target.shape == (batch_size, seq_len, execution_dim), "SystemBridge output shape mismatch!"
    print("   ✓ SystemBridge test passed!")

    # Test CrossSystemBridge
    print("\n2. Testing CrossSystemBridge...")
    cross_bridge = CrossSystemBridge(
        perception_dim=perception_dim,
        execution_dim=execution_dim,
        structure_dim=structure_dim,
        meta_dim=meta_dim,
        bridge_dim=bridge_dim
    )

    perception_state = torch.randn(batch_size, seq_len, perception_dim)
    execution_state = torch.randn(batch_size, seq_len, execution_dim)
    structure_state = torch.randn(batch_size, seq_len, structure_dim)
    meta_state = torch.randn(batch_size, seq_len, meta_dim)

    updated = cross_bridge(
        perception_state,
        execution_state,
        structure_state,
        meta_state,
        return_bridge_signals=True
    )

    p_out, e_out, s_out, m_out, bridge_signals = updated

    print(f"   Perception output: {p_out.shape}")
    print(f"   Execution output: {e_out.shape}")
    print(f"   Structure output: {s_out.shape}")
    print(f"   Meta output: {m_out.shape}")

    assert p_out.shape == perception_state.shape, "Perception shape changed!"
    assert e_out.shape == execution_state.shape, "Execution shape changed!"
    assert s_out.shape == structure_state.shape, "Structure shape changed!"
    assert m_out.shape == meta_state.shape, "Meta shape changed!"

    if bridge_signals:
        print(f"   Bridge signals: {list(bridge_signals.keys())}")

    print("   ✓ CrossSystemBridge test passed!")

    # Test BridgeSynchronizer
    print("\n3. Testing BridgeSynchronizer...")
    sync = BridgeSynchronizer(sync_interval=2, total_perception_layers=8)

    print(f"   Sync points: {sync.sync_points}")
    for layer in range(8):
        should = sync.should_sync(layer)
        print(f"   Layer {layer}: should_sync = {should}")
    assert sync.should_sync(1), "Should sync at layer 1!"
    assert sync.should_sync(3), "Should sync at layer 3!"
    assert not sync.should_sync(0), "Should not sync at layer 0!"
    print("   ✓ BridgeSynchronizer test passed!")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = p_out.sum() + e_out.sum() + s_out.sum() + m_out.sum()
    loss.backward()
    grad_exists = False
    for param in cross_bridge.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Test BridgedSystemOutput
    print("\n5. Testing BridgedSystemOutput...")
    out_proj = BridgedSystemOutput(perception_dim, 256)
    output = out_proj(perception_state)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 256), "Output shape mismatch!"
    print("   ✓ BridgedSystemOutput test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in cross_bridge.parameters())
    print(f"\n   Total parameters (CrossSystemBridge): {total_params:,}")

    print("\n" + "=" * 60)
    print("All Cross-System Bridge tests passed!")
    print("=" * 60)
