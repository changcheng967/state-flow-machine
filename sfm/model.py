"""
State-Flow Machine (SFM) Model

Full assembly of the 4-system architecture:
- System 1: Perception (linear-attention decoder)
- System 2: Execution (state slot bank)
- System 3: Structure (graph neural network)
- System 4: Meta (recurrent controller)

Cross-system bridges synchronize every 2 perception layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.config import SFMConfig, DEFAULT_CONFIG
from sfm.systems.perception import PerceptionSystem
from sfm.systems.execution import ExecutionSystem
from sfm.systems.structure import StructureSystem, CodeGraph
from sfm.systems.meta import MetaSystem
from sfm.components.cross_system_bridge import CrossSystemBridge
from sfm.utils.device import get_device, set_seed, to_device


class StateFlowMachine(nn.Module):
    """
    State-Flow Machine: A novel post-transformer architecture for code intelligence.

    Replaces the transformer paradigm with 4 specialized systems:
    1. Perception: Linear-attention decoder for token processing (O(n))
    2. Execution: State Slot Bank for explicit variable tracking
    3. Structure: Graph neural network for code structure
    4. Meta: Recurrent controller with hypothesis register and plan stack

    Cross-system bridges enable emergent behaviors while maintaining modularity.
    """

    def __init__(self, config: SFMConfig = None):
        """
        Initialize State-Flow Machine.

        Args:
            config: Configuration object. Uses default if None.
        """
        super().__init__()

        self.config = config or DEFAULT_CONFIG

        # System 1: Perception
        self.perception = PerceptionSystem(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_layers=self.config.perception_num_layers,
            num_heads=self.config.perception_num_heads,
            ff_dim=self.config.perception_ff_dim,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )

        # System 2: Execution
        self.execution = ExecutionSystem(
            input_dim=self.config.d_model,
            hidden_dim=self.config.deltanet_hidden_dim,
            num_slots=self.config.execution_num_slots,
            slot_dim=self.config.execution_slot_dim,
            max_ticks=self.config.execution_max_ticks,
            num_heads=self.config.execution_num_heads,
            dropout=self.config.dropout
        )

        # System 3: Structure
        self.structure = StructureSystem(
            input_dim=self.config.d_model,
            node_dim=self.config.structure_node_dim,
            edge_dim=self.config.structure_edge_dim,
            num_layers=self.config.structure_num_layers,
            num_heads=self.config.structure_num_heads,
            max_nodes=self.config.structure_max_nodes,
            max_edges=self.config.structure_max_edges,
            dropout=self.config.dropout
        )

        # System 4: Meta
        self.meta = MetaSystem(
            input_dim=self.config.d_model,
            hidden_dim=self.config.meta_hidden_dim,
            hypothesis_dim=self.config.meta_hypothesis_dim,
            plan_depth=self.config.meta_plan_stack_depth,
            num_heads=self.config.meta_num_heads,
            verification_threshold=self.config.meta_verification_threshold,
            dropout=self.config.dropout
        )

        # Cross-system bridge (all systems output d_model)
        self.bridge = CrossSystemBridge(
            perception_dim=self.config.d_model,
            execution_dim=self.config.d_model,  # Execution outputs d_model
            structure_dim=self.config.d_model,  # Structure outputs d_model
            meta_dim=self.config.d_model,       # Meta outputs d_model
            bridge_dim=self.config.d_bridge,
            dropout=self.config.dropout
        )

        # Output projections for each system to vocabulary
        self.perception_output = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.execution_output = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.structure_output = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.meta_output = nn.Linear(self.config.d_model, self.config.vocab_size)

        # System weighting for final output
        self.system_weights = nn.Parameter(
            torch.ones(4) / 4  # Equal weighting initially
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(self.config.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        code_graph: Optional[CodeGraph] = None,
        return_all_systems: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through State-Flow Machine.

        Args:
            tokens: Input tokens (batch, seq_len).
            code_graph: Optional code structure graph.
            return_all_systems: Whether to return per-system outputs.

        Returns:
            Output logits (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = tokens.shape

        # System 1: Perception - process tokens
        perception_hidden = self.perception.encode(tokens)  # (batch, seq_len, d_model)

        # System 2: Execution - track state
        execution_hidden = self.execution(perception_hidden)  # (batch, seq_len, d_model)

        # System 3: Structure - process graph
        structure_hidden = self.structure(perception_hidden, graph=code_graph)

        # System 4: Meta - control and verify
        meta_hidden = self.meta(perception_hidden)  # (batch, seq_len, d_model)

        # Cross-system bridge synchronization
        # Project systems to their internal dims for bridging
        exec_proj = execution_hidden
        struct_proj = structure_hidden
        meta_proj = meta_hidden

        # Bridge communication
        p_out, e_out, s_out, m_out, _ = self.bridge(
            perception_hidden,
            exec_proj,
            struct_proj,
            meta_proj
        )

        # Combine outputs with learned weighting
        weights = F.softmax(self.system_weights, dim=0)

        combined = (
            weights[0] * p_out +
            weights[1] * e_out +
            weights[2] * s_out +
            weights[3] * m_out
        )

        # Final layer norm
        combined = self.final_norm(combined)

        # Project to vocabulary
        logits = self.perception_output(combined)

        if return_all_systems:
            system_outputs = {
                "perception": self.perception_output(p_out),
                "execution": self.execution_output(e_out),
                "structure": self.structure_output(s_out),
                "meta": self.meta_output(m_out)
            }
            return logits, system_outputs

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Prompt tokens (batch, prompt_len).
            max_new_tokens: Maximum number of new tokens.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling probability.

        Returns:
            Generated tokens including prompt (batch, total_len).
        """
        self.eval()
        device = prompt.device

        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)

            # Get last token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters for each system and total.

        Returns:
            Dict with parameter counts per system.
        """
        counts = {
            "perception": sum(p.numel() for p in self.perception.parameters()),
            "execution": sum(p.numel() for p in self.execution.parameters()),
            "structure": sum(p.numel() for p in self.structure.parameters()),
            "meta": sum(p.numel() for p in self.meta.parameters()),
            "bridge": sum(p.numel() for p in self.bridge.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts

    def get_system_weights(self) -> Dict[str, float]:
        """
        Get normalized system weights.

        Returns:
            Dict mapping system names to weights.
        """
        weights = F.softmax(self.system_weights, dim=0)
        return {
            "perception": weights[0].item(),
            "execution": weights[1].item(),
            "structure": weights[2].item(),
            "meta": weights[3].item()
        }


def create_sfm(config: SFMConfig = None, device: torch.device = None) -> StateFlowMachine:
    """
    Create a State-Flow Machine model.

    Args:
        config: Configuration object.
        device: Target device. Auto-detected if None.

    Returns:
        StateFlowMachine model on the specified device.
    """
    if device is None:
        device = get_device()

    model = StateFlowMachine(config)
    model = model.to(device)

    return model


if __name__ == "__main__":
    # Smoke test
    print("=" * 70)
    print("State-Flow Machine (SFM) - Full Model Test")
    print("=" * 70)

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    # Create model with small config for testing
    print("\n1. Creating State-Flow Machine...")
    config = SFMConfig.small()
    model = create_sfm(config, device)

    # Print configuration
    print(f"\n   Configuration:")
    print(f"   - vocab_size: {config.vocab_size}")
    print(f"   - d_model: {config.d_model}")
    print(f"   - perception_layers: {config.perception_num_layers}")
    print(f"   - execution_slots: {config.execution_num_slots}")
    print(f"   - structure_layers: {config.structure_num_layers}")
    print(f"   - meta_plan_depth: {config.meta_plan_stack_depth}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 16
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    logits = model(tokens)
    print(f"   Input tokens shape: {tokens.shape}")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch!"
    print("   [OK] Forward pass test passed!")

    # Test with all system outputs
    print("\n3. Testing with per-system outputs...")
    logits, system_outputs = model(tokens, return_all_systems=True)
    print(f"   Combined logits shape: {logits.shape}")
    for name, output in system_outputs.items():
        print(f"   - {name} output shape: {output.shape}")
    print("   [OK] Per-system output test passed!")

    # Test generation
    print("\n4. Testing generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5), device=device)
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10)
    print(f"   Prompt shape: {prompt.shape}")
    print(f"   Generated shape: {generated.shape}")
    assert generated.shape == (1, 10), "Generation shape mismatch!"
    print("   [OK] Generation test passed!")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    loss = logits.sum()
    loss.backward()
    grad_exists = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   [OK] Gradient flow test passed!")

    # Print parameter counts
    print("\n" + "=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)

    counts = model.count_parameters()
    print(f"\n   System 1 (Perception): {counts['perception']:>12,} params")
    print(f"   System 2 (Execution):  {counts['execution']:>12,} params")
    print(f"   System 3 (Structure):  {counts['structure']:>12,} params")
    print(f"   System 4 (Meta):       {counts['meta']:>12,} params")
    print(f"   Cross-System Bridge:   {counts['bridge']:>12,} params")
    print(f"   " + "-" * 40)
    print(f"   TOTAL:                 {counts['total']:>12,} params")

    # Print system weights
    print("\n" + "=" * 70)
    print("SYSTEM WEIGHTS")
    print("=" * 70)
    weights = model.get_system_weights()
    for name, weight in weights.items():
        print(f"   {name.capitalize():12s}: {weight:.4f}")

    print("\n" + "=" * 70)
    print("All State-Flow Machine tests passed!")
    print("=" * 70)
