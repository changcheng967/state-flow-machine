"""
Synthetic Data Generator for State Tracking Experiment

Generates execution traces from simple programs to test state tracking ability.
Programs are sequences of variable assignments and operations.

DIFFICULTY LEVELS:
- easy: No multiplication, small values (0-50), clamped results
- medium: With multiplication, medium values (0-100), clamped results
- hard: All operations, larger values, loops, conditionals
"""

import random
from typing import List, Dict, Tuple, Optional
import json


class SimpleProgramGenerator:
    """
    Generates simple programs with clear execution traces.

    Programs consist of:
    - Variable assignments (x = 5)
    - Arithmetic operations (x = y + z)
    - Variable updates (x = x + 1)

    The goal is to track the final value of a target variable.
    """

    DIFFICULTY_CONFIGS = {
        "easy": {
            "value_range": (0, 50),
            "operators": ['+', '-'],
            "max_operand": 10,
            "clamp_values": True,
            "clamp_range": (0, 100),  # CLAMPED to [0, 100] for regression
        },
        "medium": {
            "value_range": (0, 100),
            "operators": ['+', '-', '*'],
            "max_operand": 10,
            "clamp_values": True,
            "clamp_range": (0, 100),  # CLAMPED to [0, 100] for regression
        },
        "hard": {
            "value_range": (0, 100),
            "operators": ['+', '-', '*'],
            "max_operand": 20,
            "clamp_values": True,
            "clamp_range": (0, 100),  # ALSO clamped for regression
        }
    }

    def __init__(
        self,
        num_variables: int = 5,
        value_range: Tuple[int, int] = (0, 50),
        max_program_length: int = 20,
        min_program_length: int = 3,
        seed: Optional[int] = None,
        difficulty: str = "easy"
    ):
        """
        Initialize program generator.

        Args:
            num_variables: Number of available variables.
            value_range: Range for initial values.
            max_program_length: Maximum program length.
            min_program_length: Minimum program length.
            seed: Random seed.
            difficulty: Difficulty level ("easy", "medium", "hard").
        """
        self.num_variables = num_variables
        self.max_program_length = max_program_length
        self.min_program_length = min_program_length
        self.difficulty = difficulty

        # Apply difficulty config
        config = self.DIFFICULTY_CONFIGS.get(difficulty, self.DIFFICULTY_CONFIGS["easy"])
        self.value_range = config["value_range"]
        self.operators = config["operators"]
        self.max_operand = config["max_operand"]
        self.clamp_values = config["clamp_values"]
        self.clamp_range = config["clamp_range"]

        if seed is not None:
            random.seed(seed)

        self.variables = [chr(ord('a') + i) for i in range(num_variables)]

    def _clamp(self, value: int) -> int:
        """Clamp value to valid range."""
        if self.clamp_values and self.clamp_range:
            return max(self.clamp_range[0], min(self.clamp_range[1], value))
        return value

    def generate_program(
        self,
        target_variable: Optional[str] = None
    ) -> Tuple[List[str], str, int]:
        """
        Generate a single program.

        Args:
            target_variable: Variable to track (random if None).

        Returns:
            Tuple of (program_lines, target_variable, final_value).
        """
        if target_variable is None:
            target_variable = random.choice(self.variables)

        # Initialize all variables
        state = {}
        program = []

        for var in self.variables:
            value = random.randint(*self.value_range)
            state[var] = value
            program.append(f"{var} = {value}")

        # Add operations
        num_ops = random.randint(self.min_program_length, self.max_program_length)

        for _ in range(num_ops):
            op_type = random.choice(['assign', 'update', 'copy'])

            if op_type == 'assign':
                # New assignment
                var = random.choice(self.variables)
                value = random.randint(*self.value_range)
                state[var] = self._clamp(value)
                program.append(f"{var} = {value}")

            elif op_type == 'update':
                # Update with operation
                var = random.choice(self.variables)
                other_var = random.choice(self.variables)
                op = random.choice(self.operators)
                operand = random.randint(1, self.max_operand)

                use_var = random.random() > 0.5
                current_val = state.get(var, 0)
                other_val = state.get(other_var, 0)

                if op == '+':
                    if use_var:
                        state[var] = self._clamp(current_val + other_val)
                        program.append(f"{var} = {var} + {other_var}")
                    else:
                        state[var] = self._clamp(current_val + operand)
                        program.append(f"{var} = {var} + {operand}")
                elif op == '-':
                    if use_var:
                        state[var] = self._clamp(current_val - other_val)
                        program.append(f"{var} = {var} - {other_var}")
                    else:
                        state[var] = self._clamp(current_val - operand)
                        program.append(f"{var} = {var} - {operand}")
                elif op == '*':
                    if use_var:
                        state[var] = self._clamp(current_val * other_val)
                        program.append(f"{var} = {var} * {other_var}")
                    else:
                        state[var] = self._clamp(current_val * operand)
                        program.append(f"{var} = {var} * {operand}")

            elif op_type == 'copy':
                # Copy from another variable
                var = random.choice(self.variables)
                other = random.choice(self.variables)
                state[var] = self._clamp(state.get(other, 0))
                program.append(f"{var} = {other}")

        final_value = state.get(target_variable, 0)
        # Ensure final_value is in valid class range [0, 499]
        final_value = max(0, min(499, final_value))

        # Add target query at the end
        program.append(f"# What is {target_variable}?")
        program.append(f"# Answer: {final_value}")

        return program, target_variable, final_value

    def generate_execution_trace(
        self,
        program: List[str]
    ) -> List[Dict[str, any]]:
        """
        Generate execution trace from program.

        Args:
            program: List of program lines.

        Returns:
            List of state snapshots after each line.
        """
        trace = []
        state = {}

        for line in program:
            if line.startswith('#'):
                continue

            # Parse assignment
            if '=' in line:
                parts = line.split('=')
                lhs = parts[0].strip()

                if len(parts) == 2:
                    rhs = parts[1].strip()

                    # Parse RHS
                    if rhs.isdigit():
                        state[lhs] = self._clamp(int(rhs))
                    elif rhs in state:
                        state[lhs] = self._clamp(state[rhs])
                    elif '+' in rhs:
                        # Handle addition
                        operands = rhs.split('+')
                        val = 0
                        for op in operands:
                            op = op.strip()
                            if op.isdigit():
                                val += int(op)
                            elif op in state:
                                val += state[op]
                        state[lhs] = self._clamp(val)
                    elif '-' in rhs:
                        # Handle subtraction
                        operands = rhs.split('-')
                        val = 0
                        for i, op in enumerate(operands):
                            op = op.strip()
                            if op.isdigit():
                                if i == 0:
                                    val = int(op)
                                else:
                                    val -= int(op)
                            elif op in state:
                                if i == 0:
                                    val = state[op]
                                else:
                                    val -= state[op]
                        state[lhs] = self._clamp(val)
                    elif '*' in rhs:
                        # Handle multiplication
                        operands = rhs.split('*')
                        val = 1
                        for op in operands:
                            op = op.strip()
                            if op.isdigit():
                                val *= int(op)
                            elif op in state:
                                val *= state[op]
                        state[lhs] = self._clamp(val)

            trace.append({
                "line": line,
                "state": state.copy()
            })

        return trace

    def generate_dataset(
        self,
        num_samples: int,
        include_trace: bool = True
    ) -> List[Dict[str, any]]:
        """
        Generate a dataset of programs with execution traces.

        Args:
            num_samples: Number of samples to generate.
            include_trace: Whether to include execution traces.

        Returns:
            List of samples.
        """
        dataset = []

        for _ in range(num_samples):
            program, target, final_value = self.generate_program()
            trace = self.generate_execution_trace(program) if include_trace else None

            sample = {
                "program": program,
                "target_variable": target,
                "final_value": final_value,
                "num_lines": len(program)
            }

            if include_trace:
                sample["trace"] = trace

            dataset.append(sample)

        return dataset


def generate_and_save(
    output_dir: str,
    train_samples: int = 10000,
    val_samples: int = 1000,
    max_program_length: int = 20,
    seed: int = 42,
    difficulty: str = "easy"
):
    """
    Generate and save train/val datasets.

    Args:
        output_dir: Output directory.
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        max_program_length: Maximum program length.
        seed: Random seed.
        difficulty: Difficulty level ("easy", "medium", "hard").
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    generator = SimpleProgramGenerator(
        num_variables=5,
        max_program_length=max_program_length,
        seed=seed,
        difficulty=difficulty
    )

    print(f"Generating {train_samples} training samples (difficulty: {difficulty})...")
    train_data = generator.generate_dataset(train_samples)

    print(f"Generating {val_samples} validation samples...")
    val_data = generator.generate_dataset(val_samples)

    # Save datasets
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved training data to {train_path}")

    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved validation data to {val_path}")

    # Statistics
    train_lengths = [s["num_lines"] for s in train_data]
    train_values = [s["final_value"] for s in train_data]
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Avg program length: {sum(train_lengths)/len(train_lengths):.1f}")
    print(f"  Min/Max length: {min(train_lengths)} / {max(train_lengths)}")
    print(f"  Value range: {min(train_values)} / {max(train_values)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic state tracking data")
    parser.add_argument("--output_dir", type=str, default="data/exp0",
                        help="Output directory")
    parser.add_argument("--train_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Number of validation samples")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximum program length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Difficulty level")

    args = parser.parse_args()

    generate_and_save(
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        max_program_length=args.max_length,
        seed=args.seed,
        difficulty=args.difficulty
    )
