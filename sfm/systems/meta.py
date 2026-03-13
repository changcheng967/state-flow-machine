"""
System 4: Meta

Small recurrent controller that:
1. Maintains a hypothesis register (what it thinks is wrong)
2. Maintains a plan stack (what it intends to do)
3. Has a verification head that checks its own output before emitting

Prevents death-spirals (repeated failed attempts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.components.deltanet_cell import DeltaNetCell


class HypothesisRegister(nn.Module):
    """
    Hypothesis register for tracking what the model thinks is wrong.

    Maintains a fixed-size embedding that represents the current hypothesis
    about the problem or task at hand.
    """

    def __init__(
        self,
        hidden_dim: int,
        hypothesis_dim: int,
        max_hypotheses: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize hypothesis register.

        Args:
            hidden_dim: Hidden dimension.
            hypothesis_dim: Dimension per hypothesis.
            max_hypotheses: Maximum number of hypotheses to track.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.hypothesis_dim = hypothesis_dim
        self.max_hypotheses = max_hypotheses

        # Hypothesis storage
        self.hypothesis_proj = nn.Linear(hidden_dim, hypothesis_dim)

        # Confidence scoring
        self.confidence_net = nn.Sequential(
            nn.Linear(hypothesis_dim, hypothesis_dim // 2),
            nn.ReLU(),
            nn.Linear(hypothesis_dim // 2, 1),
            nn.Sigmoid()
        )

        # Hypothesis update gate
        self.update_gate = nn.Linear(hidden_dim + hypothesis_dim, hypothesis_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hypothesis_dim)

    def init_hypotheses(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Initialize hypothesis register.

        Args:
            batch_size: Batch size.
            device: Device.
            dtype: Data type.

        Returns:
            Initial hypotheses (batch, max_hypotheses, hypothesis_dim).
        """
        return torch.zeros(batch_size, self.max_hypotheses, self.hypothesis_dim,
                          device=device, dtype=dtype)

    def forward(
        self,
        hidden_state: torch.Tensor,
        hypotheses: torch.Tensor,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Update hypothesis register.

        Args:
            hidden_state: Current hidden state (batch, hidden_dim).
            hypotheses: Current hypotheses (batch, max_hypotheses, hypothesis_dim).
            return_scores: Whether to return confidence scores.

        Returns:
            Tuple of (updated_hypotheses, primary_hypothesis, confidence_scores).
        """
        batch_size = hidden_state.size(0)

        # Project hidden state to hypothesis space
        new_hypothesis = self.hypothesis_proj(hidden_state)
        new_hypothesis = self.layer_norm(new_hypothesis)
        new_hypothesis = self.dropout(new_hypothesis)

        # Compute confidence for existing hypotheses
        confidence_scores = self.confidence_net(hypotheses).squeeze(-1)  # (batch, max_hypotheses)

        # Find least confident hypothesis to replace
        min_conf_idx = confidence_scores.argmin(dim=1)  # (batch,)

        # Update gate: how much to blend new hypothesis with old
        gate_input = torch.cat([hidden_state, new_hypothesis], dim=-1)
        gate = torch.sigmoid(self.update_gate(gate_input))  # (batch, hypothesis_dim)

        # Replace least confident hypothesis
        updated_hypotheses = hypotheses.clone()
        for i in range(batch_size):
            idx = min_conf_idx[i]
            old_hyp = hypotheses[i, idx]
            updated_hypotheses[i, idx] = gate[i] * old_hyp + (1 - gate[i]) * new_hypothesis[i]

        # Get primary hypothesis (highest confidence)
        max_conf_idx = confidence_scores.argmax(dim=1)
        primary_hypothesis = torch.stack([
            updated_hypotheses[i, max_conf_idx[i]]
            for i in range(batch_size)
        ], dim=0)

        if return_scores:
            return updated_hypotheses, primary_hypothesis, confidence_scores
        return updated_hypotheses, primary_hypothesis, None


class PlanStack(nn.Module):
    """
    Plan stack for tracking intended actions.

    Maintains a stack of planned operations that can be pushed to and popped from.
    """

    def __init__(
        self,
        hidden_dim: int,
        stack_depth: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize plan stack.

        Args:
            hidden_dim: Hidden dimension (same as plan item dimension).
            stack_depth: Maximum stack depth.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.stack_depth = stack_depth

        # Stack pointer (learnable initial bias)
        self.stack_pointer_bias = nn.Parameter(torch.zeros(1))

        # Push network
        self.push_net = nn.Linear(hidden_dim, hidden_dim)

        # Pop network (returns item and new stack pointer)
        self.pop_gate = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def init_stack(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize plan stack.

        Args:
            batch_size: Batch size.
            device: Device.
            dtype: Data type.

        Returns:
            Tuple of (stack, stack_pointer).
            - stack: (batch, stack_depth, hidden_dim)
            - stack_pointer: (batch, 1)
        """
        stack = torch.zeros(batch_size, self.stack_depth, self.hidden_dim,
                           device=device, dtype=dtype)
        pointer = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return stack, pointer

    def forward(
        self,
        hidden_state: torch.Tensor,
        stack: torch.Tensor,
        pointer: torch.Tensor,
        action: str = "auto"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Update plan stack.

        Args:
            hidden_state: Current hidden state (batch, hidden_dim).
            stack: Current stack (batch, stack_depth, hidden_dim).
            pointer: Current stack pointer (batch, 1).
            action: "push", "pop", or "auto".

        Returns:
            Tuple of (updated_stack, updated_pointer, top_item, popped_item).
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Compute push item
        push_item = self.push_net(hidden_state)
        push_item = self.layer_norm(push_item)
        push_item = self.dropout(push_item)

        # Determine action
        if action == "auto":
            pop_prob = torch.sigmoid(self.pop_gate(hidden_state))
            action_is_pop = pop_prob > 0.5
        elif action == "pop":
            action_is_pop = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        else:
            action_is_pop = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Initialize outputs
        new_stack = stack.clone()
        new_pointer = pointer.clone()
        top_item = torch.zeros_like(hidden_state)
        popped_item = None

        for i in range(batch_size):
            ptr = int(pointer[i, 0].item())

            if action_is_pop[i, 0] and ptr > 0:
                # Pop: decrement pointer and return top item
                new_pointer[i, 0] = ptr - 1
                top_item[i] = stack[i, ptr - 1]
                popped_item = stack[i, ptr - 1].unsqueeze(0) if popped_item is None else \
                              torch.cat([popped_item, stack[i, ptr - 1].unsqueeze(0)], dim=0)
            elif not action_is_pop[i, 0] and ptr < self.stack_depth:
                # Push: add item and increment pointer
                new_stack[i, ptr] = push_item[i]
                new_pointer[i, 0] = ptr + 1
                top_item[i] = push_item[i]
            else:
                # No action or invalid
                if ptr > 0:
                    top_item[i] = stack[i, ptr - 1]

        return new_stack, new_pointer, top_item, popped_item


class VerificationHead(nn.Module):
    """
    Verification head that checks output quality before emitting.

    Prevents death-spirals by rejecting low-quality outputs.
    """

    def __init__(
        self,
        hidden_dim: int,
        verification_threshold: float = 0.8,
        dropout: float = 0.1
    ):
        """
        Initialize verification head.

        Args:
            hidden_dim: Hidden dimension.
            verification_threshold: Threshold for accepting output.
            dropout: Dropout probability.
        """
        super().__init__()

        self.verification_threshold = verification_threshold

        # Quality scorer
        self.quality_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Failure detector
        self.failure_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Correction generator
        self.correction_net = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        output: torch.Tensor,
        return_correction: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Verify output quality.

        Args:
            output: Output to verify (batch, hidden_dim).
            return_correction: Whether to generate correction if needed.

        Returns:
            Tuple of (quality_score, is_accepted, correction).
        """
        # Compute quality score
        quality_score = self.quality_net(output)  # (batch, 1)

        # Detect potential failure
        failure_prob = self.failure_net(output)  # (batch, 1)

        # Combined acceptance
        is_accepted = (quality_score > self.verification_threshold) & (failure_prob < 0.5)
        is_accepted = is_accepted.float()  # (batch, 1)

        # Generate correction if needed
        correction = None
        if return_correction:
            correction = self.correction_net(output)
            correction = self.dropout(correction)

        return quality_score, is_accepted, correction


class MetaSystem(nn.Module):
    """
    System 4: Meta

    Small recurrent controller with:
    - Hypothesis register: tracks what it thinks is wrong
    - Plan stack: tracks what it intends to do
    - Verification head: checks output before emitting

    Prevents death-spirals (repeated failed attempts).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        hypothesis_dim: int = 128,
        plan_depth: int = 8,
        num_heads: int = 4,
        verification_threshold: float = 0.8,
        max_attempts: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize meta system.

        Args:
            input_dim: Input dimension from perception.
            hidden_dim: Internal hidden dimension.
            hypothesis_dim: Dimension per hypothesis.
            plan_depth: Maximum plan stack depth.
            num_heads: Number of attention heads.
            verification_threshold: Threshold for output verification.
            max_attempts: Maximum attempts before giving up.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_attempts = max_attempts

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # DeltaNet core for recurrent processing
        self.core = DeltaNetCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            eigenvalue_init=0.9,
            dropout=dropout
        )

        # Hypothesis register
        self.hypotheses = HypothesisRegister(
            hidden_dim=hidden_dim,
            hypothesis_dim=hypothesis_dim,
            max_hypotheses=4,
            dropout=dropout
        )

        # Plan stack
        self.plan_stack = PlanStack(
            hidden_dim=hidden_dim,
            stack_depth=plan_depth,
            dropout=dropout
        )

        # Verification head
        self.verification = VerificationHead(
            hidden_dim=hidden_dim,
            verification_threshold=verification_threshold,
            dropout=dropout
        )

        # Hypothesis projection (hypothesis_dim -> hidden_dim)
        self.hypothesis_proj = nn.Linear(hypothesis_dim, hidden_dim) if hypothesis_dim != hidden_dim else nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Attempt counter
        self.attempt_counter = nn.Parameter(
            torch.zeros(1),
            requires_grad=False
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize meta system state.

        Args:
            batch_size: Batch size.
            device: Device.
            dtype: Data type.

        Returns:
            State dict.
        """
        hypotheses = self.hypotheses.init_hypotheses(batch_size, device, dtype)
        stack, pointer = self.plan_stack.init_stack(batch_size, device, dtype)

        return {
            "hidden": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "hypotheses": hypotheses,
            "stack": stack,
            "pointer": pointer,
            "attempts": torch.zeros(batch_size, 1, device=device, dtype=dtype)
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through meta system.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            state: Optional previous state.
            return_state: Whether to return updated state.

        Returns:
            Output tensor, optionally with state.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device, dtype)

        hidden = state["hidden"]
        hypotheses = state["hypotheses"]
        stack = state["stack"]
        pointer = state["pointer"]
        attempts = state["attempts"]

        outputs = []
        new_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)

            # Project input
            x_proj = self.input_proj(x_t)

            # Update hidden state through DeltaNet
            hidden, _ = self.core(x_proj, hidden)

            # Update hypothesis register
            hypotheses, primary_hypothesis, _ = self.hypotheses(hidden, hypotheses)

            # Project hypothesis to hidden_dim
            primary_hypothesis_proj = self.hypothesis_proj(primary_hypothesis)

            # Update plan stack
            stack, pointer, top_plan, _ = self.plan_stack(
                hidden + primary_hypothesis_proj, stack, pointer
            )

            # Combine hidden with hypothesis and plan
            combined = hidden + 0.5 * primary_hypothesis_proj + 0.5 * top_plan

            # Verify output
            quality, is_accepted, correction = self.verification(combined, return_correction=True)

            # Apply correction if not accepted and attempts remaining
            needs_correction = (is_accepted < 0.5) & (attempts < self.max_attempts)
            corrected = torch.where(
                needs_correction,
                combined + correction,
                combined
            )

            # Update attempt counter
            attempts = torch.where(needs_correction, attempts + 1, torch.zeros_like(attempts))

            # Output projection
            output = self.output_proj(self.layer_norm(corrected))
            outputs.append(output)

            new_states.append({
                "hidden": hidden.clone(),
                "hypotheses": hypotheses.clone(),
                "stack": stack.clone(),
                "pointer": pointer.clone(),
                "attempts": attempts.clone()
            })

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, input_dim)

        # Use final state
        final_state = new_states[-1]

        if return_state:
            return output, final_state
        return output

    def get_current_hypothesis(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get current primary hypothesis.

        Args:
            state: Current state.

        Returns:
            Primary hypothesis (batch, hypothesis_dim).
        """
        hypotheses = state["hypotheses"]
        confidence = self.hypotheses.confidence_net(hypotheses).squeeze(-1)
        max_idx = confidence.argmax(dim=1)
        return torch.stack([hypotheses[i, max_idx[i]] for i in range(hypotheses.size(0))], dim=0)

    def get_plan_depth(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get current plan stack depth.

        Args:
            state: Current state.

        Returns:
            Stack depth (batch, 1).
        """
        return state["pointer"]

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Meta System Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    input_dim = 128
    hidden_dim = 128
    hypothesis_dim = 64
    plan_depth = 4

    # Initialize meta system
    print("\n1. Initializing MetaSystem...")
    meta = MetaSystem(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hypothesis_dim=hypothesis_dim,
        plan_depth=plan_depth,
        num_heads=4,
        verification_threshold=0.5  # Lower for testing
    )

    # Test HypothesisRegister
    print("\n2. Testing HypothesisRegister...")
    hidden = torch.randn(batch_size, hidden_dim)
    hypotheses = meta.hypotheses.init_hypotheses(batch_size, torch.device('cpu'), torch.float32)
    updated_hyp, primary, scores = meta.hypotheses(hidden, hypotheses, return_scores=True)
    print(f"   Hypotheses shape: {updated_hyp.shape}")
    print(f"   Primary hypothesis shape: {primary.shape}")
    print(f"   Confidence scores shape: {scores.shape}")
    assert updated_hyp.shape == (batch_size, 4, hypothesis_dim), "Hypotheses shape mismatch!"
    assert primary.shape == (batch_size, hypothesis_dim), "Primary hypothesis shape mismatch!"
    print("   ✓ HypothesisRegister test passed!")

    # Test PlanStack
    print("\n3. Testing PlanStack...")
    stack, pointer = meta.plan_stack.init_stack(batch_size, torch.device('cpu'), torch.float32)
    new_stack, new_ptr, top, popped = meta.plan_stack(hidden, stack, pointer, action="push")
    print(f"   Stack shape: {new_stack.shape}")
    print(f"   Pointer shape: {new_ptr.shape}")
    print(f"   Top item shape: {top.shape}")
    assert new_stack.shape == (batch_size, plan_depth, hidden_dim), "Stack shape mismatch!"
    print("   ✓ PlanStack test passed!")

    # Test VerificationHead
    print("\n4. Testing VerificationHead...")
    quality, accepted, correction = meta.verification(hidden, return_correction=True)
    print(f"   Quality scores shape: {quality.shape}")
    print(f"   Accepted shape: {accepted.shape}")
    print(f"   Correction shape: {correction.shape}")
    assert quality.shape == (batch_size, 1), "Quality shape mismatch!"
    assert accepted.shape == (batch_size, 1), "Accepted shape mismatch!"
    print("   ✓ VerificationHead test passed!")

    # Test full MetaSystem
    print("\n5. Testing full MetaSystem...")
    x = torch.randn(batch_size, seq_len, input_dim)
    output, state = meta.forward(x, return_state=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    assert "hidden" in state, "State missing hidden!"
    assert "hypotheses" in state, "State missing hypotheses!"
    print("   ✓ MetaSystem test passed!")

    # Test state operations
    print("\n6. Testing state operations...")
    current_hyp = meta.get_current_hypothesis(state)
    plan_depth = meta.get_plan_depth(state)
    print(f"   Current hypothesis shape: {current_hyp.shape}")
    print(f"   Plan depth shape: {plan_depth.shape}")
    print(f"   Average plan depth: {plan_depth.mean().item():.2f}")
    print("   ✓ State operations test passed!")

    # Test with continuing state
    print("\n7. Testing continuing state...")
    x2 = torch.randn(batch_size, seq_len, input_dim)
    output2, state2 = meta.forward(x2, state=state, return_state=True)
    print(f"   Second output shape: {output2.shape}")
    assert output2.shape == x2.shape, "Second output shape mismatch!"
    print("   ✓ Continuing state test passed!")

    # Test gradient flow
    print("\n8. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for param in meta.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = meta.count_parameters()
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All Meta System tests passed!")
    print("=" * 60)
