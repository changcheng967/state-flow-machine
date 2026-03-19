"""gen_pillar1_traces.py — Synthetic Python execution traces.

Generates Python programs with variable assignments, arithmetic, control flow,
lists, and functions. Each program includes inline state comments showing
variable values after each statement.

Output: JSONL with "text" field.

Usage:
    python gen_pillar1_traces.py --output pillar1.jsonl --samples 500000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


# Template pool for different program types
_VAR_NAMES = ["x", "y", "z", "a", "b", "c", "d", "e",
              "n", "m", "i", "j", "k", "total", "result",
              "count", "sum_val", "tmp", "val", "data"]


def _pick_vars(n: int, exclude: set[str] | None = None) -> list[str]:
    """Pick n unique variable names."""
    exclude = exclude or set()
    pool = [v for v in _VAR_NAMES if v not in exclude]
    return random.sample(pool, min(n, len(pool)))


def gen_arithmetic(depth: int = 5) -> str:
    """Generate a sequence of arithmetic assignments with state comments."""
    vars_used: set[str] = set()
    lines = ["# Arithmetic operations"]

    for _ in range(depth):
        available = _pick_vars(2, vars_used)
        if len(available) < 2:
            available = _pick_vars(2)
        target, src = available[0], available[1]

        vars_used.add(target)
        if src not in vars_used:
            # Initialize src first
            val = random.randint(-100, 100)
            lines.append(f"{src} = {val}")
            lines.append(f"# State: {src} = {val}")
            vars_used.add(src)

        ops = ["+", "-", "*"]
        op = random.choice(ops)
        operand_type = random.choice(["literal", "variable"])

        if operand_type == "literal":
            operand = random.randint(-50, 50)
            lines.append(f"{target} = {src} {op} {operand}")
            # Compute result
            if op == "+":
                result = f"{src} + {operand}"
            elif op == "-":
                result = f"{src} - {operand}"
            else:
                result = f"{src} * {operand}"
        else:
            others = _pick_vars(3, vars_used)
            op_var = others[0] if others else "x"
            lines.append(f"{target} = {src} {op} {op_var}")
            result = f"{src} {op} {op_var}"

        lines.append(f"# State: {target} = {result}")

    lines.append(f"# Final state: {vars_used}")
    return "\n".join(lines)


def gen_control_flow(depth: int = 3) -> str:
    """Generate if/elif/else and while/for loops with state tracking."""
    vars_used: set[str] = set()
    lines = ["# Control flow"]
    target, _ = _pick_vars(2)
    vars_used.add(target)
    init_val = random.randint(0, 20)
    lines.append(f"{target} = {init_val}")
    lines.append(f"# State: {target} = {init_val}")

    for _ in range(depth):
        pattern = random.choice(["if_else", "while", "for_range", "nested"])

        if pattern == "if_else":
            comp_ops = [">", "<", ">=", "<=", "==", "!="]
            comp = random.choice(comp_ops)
            threshold = random.randint(0, 50)
            lines.append(f"if {target} {comp} {threshold}:")
            action_val = random.randint(1, 10)
            act_ops = ["+", "-", "*"]
            act_op = random.choice(act_ops)
            lines.append(f"    {target} = {target} {act_op} {action_val}")
            lines.append(f"# State: {target} = {target} {act_op} {action_val}")
            lines.append("else:")
            action_val2 = random.randint(1, 10)
            lines.append(f"    {target} = {target} - {action_val2}")
            lines.append(f"# State: {target} = {target} - {action_val2}")

        elif pattern == "while":
            threshold = random.randint(5, 30)
            update_val = random.randint(1, 5)
            lines.append(f"while {target} < {threshold}:")
            lines.append(f"    {target} = {target} + {update_val}")
            lines.append(f"# State: {target} increments by {update_val} until >= {threshold}")

        elif pattern == "for_range":
            loop_var = _pick_vars(1, vars_used)[0]
            end_val = random.randint(5, 20)
            lines.append(f"for {loop_var} in range({end_val}):")
            lines.append(f"    {target} = {target} + {loop_var}")
            lines.append(f"# State: {target} accumulates 0+1+2+...+{end_val - 1} = {(end_val - 1) * end_val // 2}")

        elif pattern == "nested":
            outer_var = _pick_vars(1, vars_used)[0]
            inner_var = _pick_vars(1, vars_used | {outer_var})[0]
            lines.append(f"for {outer_var} in range(3):")
            lines.append(f"    for {inner_var} in range(4):")
            lines.append(f"        {target} = {target} + {outer_var} * {inner_var}")

    lines.append(f"# Final state: {target}")
    return "\n".join(lines)


def gen_list_ops(depth: int = 5) -> str:
    """Generate list operations: creation, append, indexing, slicing."""
    lines = ["# List operations"]
    var_name = random.choice(["data", "items", "nums", "values", "arr", "lst"])
    init_items = random.sample(range(-20, 21), random.randint(3, 8))
    lines.append(f"{var_name} = {init_items}")
    lines.append(f"# State: {var_name} = {init_items}")

    for _ in range(depth):
        op = random.choice(["append", "pop", "sort", "reverse", "index",
                            "slice", "extend", "insert", "remove", "sum"])

        if op == "append":
            val = random.randint(-10, 10)
            lines.append(f"{var_name}.append({val})")
            lines.append(f"# State: {var_name} now ends with {val}")

        elif op == "pop":
            lines.append(f"{var_name}.pop()")
            lines.append(f"# State: {var_name} last element removed")

        elif op == "sort":
            lines.append(f"{var_name}.sort()")
            lines.append(f"# State: {var_name} sorted ascending")

        elif op == "reverse":
            lines.append(f"{var_name}.reverse()")
            lines.append(f"# State: {var_name} reversed")

        elif op == "slice":
            idx1 = random.randint(0, len(init_items) - 2)
            idx2 = random.randint(idx1 + 1, len(init_items))
            target = random.choice(["sub", "part", "section"])
            lines.append(f"{target} = {var_name}[{idx1}:{idx2}]")
            lines.append(f"# State: {target} = slice [{idx1}:{idx2}]")

        elif op == "sum":
            target = random.choice(["total", "s", "total_sum"])
            lines.append(f"{target} = sum({var_name})")
            lines.append(f"# State: {target} = sum of {var_name}")

        elif op == "insert":
            idx = random.randint(0, max(1, len(init_items) - 1))
            val = random.randint(-10, 10)
            lines.append(f"{var_name}.insert({idx}, {val})")
            lines.append(f"# State: {var_name}[{idx}] = {val}")

        elif op == "remove":
            if init_items:
                val = random.choice(init_items)
                lines.append(f"{var_name}.remove({val})")
                lines.append(f"# State: first occurrence of {val} removed")

        elif op == "extend":
            ext = random.sample(range(-10, 11), random.randint(2, 4))
            lines.append(f"{var_name}.extend({ext})")
            lines.append(f"# State: {var_name} extended by {ext}")

        elif op == "index":
            target = random.choice(["pos", "idx", "position"])
            lines.append(f"{target} = {var_name}.index({init_items[0]})")
            lines.append(f"# State: {target} = position of {init_items[0]}")

    return "\n".join(lines)


def gen_functions(depth: int = 3) -> str:
    """Generate function definitions with various patterns."""
    lines = ["# Function definitions"]

    for _ in range(depth):
        fn_name = random.choice(["compute", "process", "transform", "calc",
                                 "evaluate", "update", "apply", "solve"])
        n_params = random.randint(1, 4)
        params = _pick_vars(n_params)
        ret_var = random.choice(["result", "out", "res", "ans"])

        pattern = random.choice(["simple_return", "conditional", "recursive",
                                  "accumulator", "composition"])

        if pattern == "simple_return":
            ops = ["+", "-", "*", "//"]
            op = random.choice(ops)
            lines.append(f"def {fn_name}({', '.join(params)}):")
            lines.append(f"    {ret_var} = {params[0]} {op} {params[1] if len(params) > 1 else 1}")
            lines.append(f"    return {ret_var}")

        elif pattern == "conditional":
            lines.append(f"def {fn_name}({', '.join(params)}):")
            lines.append(f"    if {params[0]} > 0:")
            lines.append(f"        return {params[0]} * 2")
            lines.append(f"    elif {params[0]} == 0:")
            lines.append(f"        return 0")
            lines.append(f"    else:")
            lines.append(f"        return {params[0]} - 1")

        elif pattern == "recursive":
            lines.append(f"def {fn_name}({', '.join(params)}):")
            lines.append(f"    if {params[0]} <= 1:")
            lines.append(f"        return 1")
            lines.append(f"    return {params[0]} + {fn_name}({params[0]} - 1)")

        elif pattern == "accumulator":
            loop_var = random.choice(["i", "j", "k", "n"])
            lines.append(f"def {fn_name}({', '.join(params)}):")
            lines.append(f"    {ret_var} = 0")
            lines.append(f"    for {loop_var} in range({params[0]}):")
            lines.append(f"        {ret_var} += {loop_var}")
            lines.append(f"    return {ret_var}")

        elif pattern == "composition":
            fn2 = random.choice(["helper", "inner", "sub", "util"])
            lines.append(f"def {fn2}({params[0]}):")
            lines.append(f"    return {params[0]} ** 2")
            lines.append(f"")
            lines.append(f"def {fn_name}({', '.join(params)}):")
            lines.append(f"    return {fn2}({params[0]}) + {params[1] if len(params) > 1 else 0}")

        # Add usage example
        args = [str(random.randint(1, 20)) for _ in params]
        lines.append(f"")
        lines.append(f"# Example: {fn_name}({', '.join(args)})")
        lines.append(f"{ret_var} = {fn_name}({', '.join(args)})")
        lines.append(f"# State: {ret_var} = {fn_name}({', '.join(args)})")

    return "\n".join(lines)


def gen_mixed_program(seed: int | None = None) -> str:
    """Generate a random mixed-type program."""
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    generators = [gen_arithmetic, gen_control_flow, gen_list_ops, gen_functions]
    weights = [0.30, 0.25, 0.20, 0.25]
    chosen = random.choices(generators, weights=weights, k=1)[0]
    depth = random.randint(3, 8)
    return chosen(depth=depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pillar 1: execution traces")
    parser.add_argument("--output", default="pillar1.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} execution trace samples...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = gen_mixed_program()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples}", file=sys.stderr)

    print(f"Done: {args.samples} samples -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
