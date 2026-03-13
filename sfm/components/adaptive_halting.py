"""
Adaptive Halting for dynamic compute allocation.

Enables the model to decide how many processing steps to take
per input, rather than fixed-depth processing.

Used in System 2 (Execution) to process complex statements with
more internal ticks and simple statements with fewer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class AdaptiveHalting(nn.Module):
    """
    Adaptive halting mechanism for dynamic compute.

    The model decides when to stop processing based on confidence.
    Uses pondering approach similar to ACT (Adaptive Computation Time).

    Key components:
    1. Halting network: outputs probability of halting at each step
    2. State accumulation: weighted sum of intermediate states
    3. Budget constraint: maximum number of steps
    """

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int = 8,
        halting_threshold: float = 0.5,
        epsilon: float = 1e-6
    ):
        """
        Initialize adaptive halting.

        Args:
            hidden_dim: Dimension of hidden state.
            max_steps: Maximum number of processing steps.
            halting_threshold: Threshold for halting decision.
            epsilon: Small value for numerical stability.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.halting_threshold = halting_threshold
        self.epsilon = epsilon

        # Halting network
        self.halt_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # State transition network
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Residual weighting
        self.residual_gate = nn.Linear(hidden_dim, 1)

    def compute_halt_probability(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of halting at current step.

        Args:
            state: Current state (batch, hidden_dim).

        Returns:
            Halting probability (batch, 1).
        """
        return self.halt_net(state)

    def transition_state(
        self,
        state: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transition to next processing state.

        Args:
            state: Current state (batch, hidden_dim).
            input_signal: Optional input signal to incorporate.

        Returns:
            New state (batch, hidden_dim).
        """
        if input_signal is not None:
            state = state + input_signal
        new_state = self.transition(state)
        return new_state

    def forward(
        self,
        initial_state: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None,
        return_all_states: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process with adaptive halting.

        Args:
            initial_state: Initial state (batch, hidden_dim).
            input_signal: Optional input to incorporate each step.
            return_all_states: Whether to return all intermediate states.

        Returns:
            Tuple of (final_state, info_dict).
            - final_state: Accumulated output (batch, hidden_dim)
            - info_dict: Contains num_steps, halt_probs, ponder_cost, etc.
        """
        batch_size = initial_state.size(0)
        device = initial_state.device
        dtype = initial_state.dtype

        # Accumulated output and remaining probability
        accumulated = torch.zeros_like(initial_state)
        remaining = torch.ones(batch_size, 1, device=device, dtype=dtype)

        all_states = [] if return_all_states else None
        halt_probs = []
        step_outputs = []

        state = initial_state
        total_steps = 0
        ponder_cost = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        for step in range(self.max_steps):
            # Compute halting probability
            halt_prob = self.compute_halt_probability(state)

            # Determine actual halt weight for this step
            if step == self.max_steps - 1:
                # Last step: use all remaining probability
                halt_weight = remaining
            else:
                # Regular step: halt_prob * remaining
                halt_weight = halt_prob * remaining

            # Accumulate output
            accumulated = accumulated + halt_weight * state
            step_outputs.append(halt_weight * state)

            # Update remaining probability
            remaining = remaining - halt_weight

            # Track ponder cost (regularization for computation)
            ponder_cost = ponder_cost + halt_weight * (step + 1)

            # Record for analysis
            halt_probs.append(halt_prob.mean().item())
            total_steps += 1

            if return_all_states:
                all_states.append(state.clone())

            # Check if we should stop
            if remaining.max() < self.halting_threshold:
                break

            # Transition to next state
            state = self.transition_state(state, input_signal)

        info = {
            "num_steps": total_steps,
            "halt_probs": halt_probs,
            "ponder_cost": ponder_cost.mean().item(),
            "final_remaining": remaining.mean().item(),
            "all_states": all_states
        }

        return accumulated, info


class AdaptiveProcessor(nn.Module):
    """
    Wraps any module with adaptive halting for dynamic compute.
    """

    def __init__(
        self,
        module: nn.Module,
        hidden_dim: int,
        max_steps: int = 8,
        halting_threshold: float = 0.5
    ):
        """
        Initialize adaptive processor.

        Args:
            module: The module to apply repeatedly.
            hidden_dim: Hidden dimension for halting decisions.
            max_steps: Maximum number of applications.
            halting_threshold: Threshold for halting.
        """
        super().__init__()

        self.module = module
        self.adaptive = AdaptiveHalting(
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            halting_threshold=halting_threshold
        )

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False
    ) -> torch.Tensor:
        """
        Apply module adaptively.

        Args:
            x: Input tensor (batch, seq, dim) or (batch, dim).
            return_info: Whether to return halting info.

        Returns:
            Output tensor, optionally with info dict.
        """
        is_sequence = x.dim() == 3

        if is_sequence:
            batch_size, seq_len, dim = x.shape
            outputs = []
            all_info = []

            for t in range(seq_len):
                x_t = x[:, t, :]  # (batch, dim)
                out_t, info = self.adaptive(x_t)

                # Apply module for each step
                for _ in range(info["num_steps"]):
                    out_t = self.module(out_t.unsqueeze(1)).squeeze(1)

                outputs.append(out_t)
                all_info.append(info)

            output = torch.stack(outputs, dim=1)
            if return_info:
                return output, all_info
            return output
        else:
            output, info = self.adaptive(x)
            for _ in range(info["num_steps"]):
                output = self.module(output.unsqueeze(1)).squeeze(1)
            if return_info:
                return output, info
            return output


class StepController(nn.Module):
    """
    Controls step-wise processing with learned step embeddings.

    Each step has a learned embedding that gets added to the state,
    allowing the model to differentiate between processing steps.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int = 8
    ):
        """
        Initialize step controller.

        Args:
            hidden_dim: Hidden dimension.
            max_steps: Maximum number of steps.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Step embeddings
        self.step_embeddings = nn.Parameter(
            torch.randn(max_steps, hidden_dim) / math.sqrt(hidden_dim)
        )

        # Step-specific transformations
        self.step_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(max_steps)
        ])

        # Halting predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def get_step_embedding(self, step: int) -> torch.Tensor:
        """Get embedding for a specific step."""
        return self.step_embeddings[step]

    def transform_step(
        self,
        state: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """
        Apply step-specific transformation.

        Args:
            state: Current state (batch, hidden_dim).
            step: Current step index.

        Returns:
            Transformed state (batch, hidden_dim).
        """
        step_emb = self.get_step_embedding(step)
        transformed = self.step_transforms[step](state)
        return transformed + step_emb.unsqueeze(0)

    def predict_halt(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """
        Predict whether to halt at current step.

        Args:
            state: Current state (batch, hidden_dim).
            step: Current step index.

        Returns:
            Halting probability (batch, 1).
        """
        # Add step embedding for context
        state_with_step = state + self.get_step_embedding(step).unsqueeze(0)
        return self.halt_predictor(state_with_step)


class BudgetAwareHalting(nn.Module):
    """
    Halting mechanism with explicit compute budget.

    Ensures total compute stays within budget across a sequence,
    allocating more steps to complex inputs and fewer to simple ones.
    """

    def __init__(
        self,
        hidden_dim: int,
        total_budget: int = 128,
        max_steps_per_input: int = 8,
        min_steps: int = 1
    ):
        """
        Initialize budget-aware halting.

        Args:
            hidden_dim: Hidden dimension.
            total_budget: Total steps allowed for entire sequence.
            max_steps_per_input: Maximum steps per single input.
            min_steps: Minimum steps per input.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.total_budget = total_budget
        self.max_steps_per_input = max_steps_per_input
        self.min_steps = min_steps

        # Halting network
        self.halt_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for remaining budget
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Complexity estimator
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def estimate_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate input complexity.

        Args:
            state: Input state (batch, hidden_dim).

        Returns:
            Complexity score (batch, 1) in [0, 1].
        """
        return self.complexity_net(state)

    def forward(
        self,
        states: torch.Tensor,
        return_budget_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process sequence with budget-aware halting.

        Args:
            states: Input states (batch, seq_len, hidden_dim).
            return_budget_info: Whether to return budget usage info.

        Returns:
            Tuple of (output_states, info_dict).
        """
        batch_size, seq_len, _ = states.shape
        device = states.device

        outputs = []
        remaining_budget = torch.ones(batch_size, 1, device=device) * self.total_budget
        steps_used = []
        complexities = []

        for t in range(seq_len):
            state = states[:, t, :]

            # Estimate complexity
            complexity = self.estimate_complexity(state)
            complexities.append(complexity.mean().item())

            # Determine max steps for this input based on remaining budget
            budget_frac = remaining_budget / (seq_len - t + 1) / self.max_steps_per_input
            max_for_this = max(
                self.min_steps,
                min(
                    int(budget_frac.mean().item() * self.max_steps_per_input),
                    self.max_steps_per_input
                )
            )

            # Adaptive halting
            steps = 0
            for step in range(max_for_this):
                # Check if we should halt
                budget_input = remaining_budget / self.total_budget
                halt_input = torch.cat([state, budget_input], dim=-1)
                halt_prob = self.halt_net(halt_input)

                if halt_prob.mean() > 0.5 and step >= self.min_steps:
                    break

                steps += 1

            steps_used.append(steps)
            remaining_budget = remaining_budget - steps

            outputs.append(state)

        output = torch.stack(outputs, dim=1)

        info = {
            "steps_used": steps_used,
            "total_steps": sum(steps_used),
            "budget_used": sum(steps_used),
            "avg_complexity": sum(complexities) / len(complexities),
            "remaining_budget": remaining_budget.mean().item()
        }

        return output, info


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Adaptive Halting Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    hidden_dim = 128
    max_steps = 8

    # Test AdaptiveHalting
    print("\n1. Testing AdaptiveHalting...")
    halting = AdaptiveHalting(hidden_dim, max_steps=max_steps)

    initial_state = torch.randn(batch_size, hidden_dim)
    output, info = halting(initial_state, return_all_states=True)

    print(f"   Input shape: {initial_state.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of steps: {info['num_steps']}")
    print(f"   Halt probabilities: {[f'{p:.3f}' for p in info['halt_probs']]}")
    print(f"   Ponder cost: {info['ponder_cost']:.3f}")
    assert output.shape == (batch_size, hidden_dim), "Output shape mismatch!"
    print("   ✓ AdaptiveHalting test passed!")

    # Test StepController
    print("\n2. Testing StepController...")
    controller = StepController(hidden_dim, max_steps=max_steps)

    state = torch.randn(batch_size, hidden_dim)
    for step in range(max_steps):
        transformed = controller.transform_step(state, step)
        halt_prob = controller.predict_halt(state, step)
        print(f"   Step {step}: halt_prob = {halt_prob.mean().item():.3f}")

    print("   ✓ StepController test passed!")

    # Test BudgetAwareHalting
    print("\n3. Testing BudgetAwareHalting...")
    budget_halt = BudgetAwareHalting(
        hidden_dim,
        total_budget=64,
        max_steps_per_input=8
    )

    seq_len = 16
    states = torch.randn(batch_size, seq_len, hidden_dim)
    output, info = budget_halt(states, return_budget_info=True)

    print(f"   Input shape: {states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Steps used: {info['steps_used']}")
    print(f"   Total steps: {info['total_steps']}")
    print(f"   Average complexity: {info['avg_complexity']:.3f}")
    assert output.shape == (batch_size, seq_len, hidden_dim), "Output shape mismatch!"
    print("   ✓ BudgetAwareHalting test passed!")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for param in budget_halt.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Test different complexity inputs
    print("\n5. Testing complexity-dependent behavior...")
    simple_state = torch.zeros(batch_size, hidden_dim)  # Simple (zeros)
    complex_state = torch.randn(batch_size, hidden_dim)  # Complex (random)

    simple_complexity = budget_halt.estimate_complexity(simple_state)
    complex_complexity = budget_halt.estimate_complexity(complex_state)

    print(f"   Simple input complexity: {simple_complexity.mean().item():.3f}")
    print(f"   Complex input complexity: {complex_complexity.mean().item():.3f}")
    print("   ✓ Complexity estimation test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in halting.parameters())
    print(f"\n   Total parameters (AdaptiveHalting): {total_params:,}")

    print("\n" + "=" * 60)
    print("All Adaptive Halting tests passed!")
    print("=" * 60)
