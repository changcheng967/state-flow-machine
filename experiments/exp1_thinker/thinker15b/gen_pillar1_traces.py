"""gen_pillar1_traces.py — Real execution traces via exec() + namespace inspection.

v2: Actually executes generated Python code in a sandbox, inspects the
namespace after each statement, and formats output with <code>/<trace>/<slots>
XML tags. This gives the model real ground-truth state transitions.

Output: JSONL with "text" field.

Usage:
    python gen_pillar1_traces.py --output pillar1.jsonl --samples 500000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback


_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "sorted": sorted, "reversed": reversed,
    "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
    "round": round, "pow": pow, "divmod": divmod, "isinstance": isinstance,
    "type": type, "hex": hex, "bin": bin, "oct": oct, "chr": chr, "ord": ord,
    "all": all, "any": any, "print": lambda *a, **k: None,
    "True": True, "False": False, "None": None,
}


def _safe_exec(code: str, ns: dict) -> dict | None:
    """Execute code in sandboxed namespace. Returns final state or None."""
    try:
        exec(code, {"__builtins__": _SAFE_BUILTINS}, ns)
        return ns
    except Exception:
        return None


def _extract_vars(ns: dict, exclude: set[str] | None = None) -> dict:
    """Extract variable values from namespace (exclude dunders and builtins)."""
    exclude = exclude or set()
    result = {}
    for k, v in ns.items():
        if k.startswith("_") or k in exclude:
            continue
        if callable(v) or isinstance(v, type):
            continue
        # Only include simple types
        if isinstance(v, (int, float, str, bool)) or (
            isinstance(v, (list, tuple)) and len(v) < 20 and
            all(isinstance(x, (int, float, str, bool)) for x in v)
        ):
            result[k] = v
    return result


def _state_to_slots(state: dict, max_slots: int = 8) -> str:
    """Convert state dict to slot format string."""
    if not state:
        return "<slots/>"
    slot_lines = []
    for i, (k, v) in enumerate(sorted(state.items())):
        if i >= max_slots:
            slot_lines.append(f"  ... and {len(state) - max_slots} more")
            break
        v_str = repr(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        slot_lines.append(f"  {k} = {v_str}")
    return "<slots>\n" + "\n".join(slot_lines) + "\n</slots>"


def gen_arithmetic_trace() -> str:
    """Generate arithmetic assignment traces with real execution."""
    var_pool = ["x", "y", "z", "a", "b", "c", "n", "m", "total", "result",
                "val", "tmp", "s"]
    used: set[str] = set()
    num_stmts = random.randint(4, 10)

    code_lines = []
    trace_parts = ["<code>"]

    for _ in range(num_stmts):
        target = random.choice([v for v in var_pool if v not in used] or var_pool)
        used.add(target)

        # Choose operation
        op = random.choice(["+", "-", "*", "//", "%"])

        if random.random() < 0.5 and used - {target}:
            src = random.choice(list(used - {target}))
            rhs = f"{src} {op} {random.randint(1, 50)}"
        else:
            a, b = random.randint(1, 100), random.randint(1, 100)
            rhs = f"{a} {op} {b}"

        code_lines.append(f"{target} = {rhs}")
        trace_parts.append(f"  {target} = {rhs}")

    trace_parts.append("</code>")

    # Execute and trace
    ns: dict = {}
    ns_copy: dict = {}
    trace_parts.append("<trace>")
    for stmt in code_lines:
        ns_copy = dict(ns)
        _safe_exec(stmt, ns)
        new_state = _extract_vars(ns)
        old_state = _extract_vars(ns_copy)
        changed = {k: v for k, v in new_state.items()
                   if k not in old_state or old_state[k] != v}
        if changed:
            for k, v in sorted(changed.items()):
                trace_parts.append(f"  {k}: {repr(v)}")
    trace_parts.append("</trace>")

    # Final slots
    final = _extract_vars(ns)
    trace_parts.append(_state_to_slots(final))

    return "\n".join(trace_parts)


def gen_control_flow_trace() -> str:
    """Generate control flow traces (if/else, while, for) with real execution."""
    var_pool = ["x", "y", "i", "j", "n", "count", "total", "result", "flag"]
    target = random.choice(var_pool[:4])
    init_val = random.randint(0, 30)

    pattern = random.choice(["if_else", "while_loop", "for_range"])

    code_lines = [f"{target} = {init_val}"]
    trace_parts = ["<code>"]
    trace_parts.append(f"  {target} = {init_val}")

    if pattern == "if_else":
        threshold = random.randint(5, 40)
        action = random.randint(1, 10)
        code_lines.append(f"if {target} >= {threshold}:")
        trace_parts.append(f"  if {target} >= {threshold}:")
        code_lines.append(f"    {target} = {target} + {action}")
        trace_parts.append(f"    {target} = {target} + {action}")
        code_lines.append(f"else:")
        trace_parts.append(f"  else:")
        code_lines.append(f"    {target} = {target} - {action}")
        trace_parts.append(f"    {target} = {target} - {action}")

    elif pattern == "while_loop":
        limit = random.randint(10, 50)
        step = random.randint(1, 5)
        code_lines.append(f"while {target} < {limit}:")
        trace_parts.append(f"  while {target} < {limit}:")
        code_lines.append(f"    {target} = {target} + {step}")
        trace_parts.append(f"    {target} = {target} + {step}")

    elif pattern == "for_range":
        end = random.randint(5, 20)
        loop_var = random.choice([v for v in var_pool if v != target])
        code_lines.append(f"total = 0")
        trace_parts.append(f"  total = 0")
        code_lines.append(f"for {loop_var} in range({end}):")
        trace_parts.append(f"  for {loop_var} in range({end}):")
        code_lines.append(f"    total = total + {loop_var}")
        trace_parts.append(f"    total = total + {loop_var}")
        code_lines.append(f"{target} = total")
        trace_parts.append(f"  {target} = total")

    trace_parts.append("</code>")

    # Execute and trace
    ns: dict = {}
    trace_parts.append("<trace>")
    for stmt in code_lines:
        old = _extract_vars(ns)
        _safe_exec(stmt, ns)
        new = _extract_vars(ns)
        changed = {k: v for k, v in new.items()
                   if k not in old or old[k] != v}
        if changed:
            for k, v in sorted(changed.items()):
                trace_parts.append(f"  {k}: {repr(v)}")
    trace_parts.append("</trace>")

    final = _extract_vars(ns)
    trace_parts.append(_state_to_slots(final))
    return "\n".join(trace_parts)


def gen_list_ops_trace() -> str:
    """Generate list operation traces with real execution."""
    var_name = random.choice(["data", "items", "nums", "values", "arr"])
    init = sorted(random.sample(range(-20, 21), random.randint(3, 8)))
    code_lines = [f"{var_name} = {init}"]
    trace_parts = ["<code>"]
    trace_parts.append(f"  {var_name} = {init}")

    ops_pool = [
        lambda: f"{var_name}.append({random.randint(-10, 10)})",
        lambda: f"{var_name}.sort()" if init else "pass",
        lambda: f"{var_name}.reverse()" if len(init) > 1 else "pass",
        lambda: f"{var_name}.insert({random.randint(0, max(1, len(init)-1))}, {random.randint(-5, 5)})",
        lambda: f"total = sum({var_name})",
        lambda: f"{var_name}.extend({random.sample(range(-10, 11), random.randint(2, 4))})",
        lambda: f"sliced = {var_name}[{random.randint(0, max(0, len(init)-2))}:{random.randint(1, len(init))}]",
        lambda: f"result = sorted({var_name}, reverse=True)",
    ]

    num_ops = random.randint(3, 6)
    for _ in range(num_ops):
        stmt = random.choice(ops_pool)()
        if stmt != "pass":
            code_lines.append(stmt)
            trace_parts.append(f"  {stmt}")

    trace_parts.append("</code>")

    ns: dict = {}
    trace_parts.append("<trace>")
    for stmt in code_lines:
        old = _extract_vars(ns)
        _safe_exec(stmt, ns)
        new = _extract_vars(ns)
        changed = {k: v for k, v in new.items()
                   if k not in old or old[k] != v}
        for k, v in sorted(changed.items()):
            v_str = repr(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            trace_parts.append(f"  {k}: {v_str}")
    trace_parts.append("</trace>")

    final = _extract_vars(ns)
    trace_parts.append(_state_to_slots(final))
    return "\n".join(trace_parts)


def gen_function_trace() -> str:
    """Generate function definition + call with real execution and tracing."""
    fn_name = random.choice(["add", "compute", "calc", "process", "transform",
                             "solve", "evaluate", "update"])
    params = random.sample(["a", "b", "x", "y", "n", "val"], random.randint(1, 3))

    patterns = [
        # simple arithmetic
        lambda: (
            f"def {fn_name}({', '.join(params)}):\n"
            f"    return {params[0]} + {params[1] if len(params) > 1 else 0}\n",
            [random.randint(1, 50) for _ in params],
        ),
        # conditional
        lambda: (
            f"def {fn_name}({', '.join(params)}):\n"
            f"    if {params[0]} > 10:\n"
            f"        return {params[0]} * 2\n"
            f"    return {params[0]} + 5\n",
            [random.randint(1, 30) for _ in params],
        ),
        # accumulator loop
        lambda: (
            f"def {fn_name}({params[0]}):\n"
            f"    result = 0\n"
            f"    for i in range({params[0]}):\n"
            f"        result = result + i\n"
            f"    return result\n",
            [random.randint(1, 20)],
        ),
    ]

    code, args = random.choice(patterns)()
    call = f"answer = {fn_name}({', '.join(str(a) for a in args)})"

    trace_parts = ["<code>"]
    for line in code.splitlines():
        trace_parts.append(f"  {line}")
    trace_parts.append(f"  {call}")
    trace_parts.append("</code>")

    ns: dict = {}
    trace_parts.append("<trace>")
    # Execute function def
    _safe_exec(code, ns)
    trace_parts.append(f"  [function {fn_name} defined]")
    # Execute call
    _safe_exec(call, ns)
    if "answer" in ns:
        trace_parts.append(f"  answer = {repr(ns['answer'])}")
    trace_parts.append("</trace>")

    final = _extract_vars(ns)
    trace_parts.append(_state_to_slots(final))
    return "\n".join(trace_parts)


def gen_mixed_trace(seed: int | None = None) -> str:
    """Generate a random execution trace."""
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    generators = [gen_arithmetic_trace, gen_control_flow_trace,
                  gen_list_ops_trace, gen_function_trace]
    weights = [0.30, 0.25, 0.20, 0.25]
    chosen = random.choices(generators, weights=weights, k=1)[0]
    return chosen()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pillar 1: execution traces (exec-based)")
    parser.add_argument("--output", default="pillar1.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} execution trace samples...", file=sys.stderr)

    errors = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            try:
                text = gen_mixed_trace()
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error at sample {i}: {e}", file=sys.stderr)
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples} (errors: {errors})",
                      file=sys.stderr)

    print(f"Done: {args.samples} samples, {errors} errors -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
