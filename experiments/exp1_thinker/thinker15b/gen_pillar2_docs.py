"""gen_pillar2_docs.py — Synthetic API documentation.

Generates fictional but realistic API documentation: module docs,
class definitions, function signatures with type annotations, docstrings,
and usage examples.

Output: JSONL with "text" field.

Usage:
    python gen_pillar2_docs.py --output pillar2.jsonl --samples 500000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


_MODULE_PREFIXES = [
    "dataflow", "neural", "quantum", "vector", "tensor",
    "asyncio_ext", "crypto_lib", "graph_db", "stream_proc",
    "cache_layer", "auth_middleware", "event_bus", "task_queue",
    "ml_pipeline", "config_store", "log_aggregator", "metric_collector",
    "rate_limiter", "circuit_breaker", "service_mesh", "api_gateway",
]

_CLASS_NAMES = [
    "DataFrame", "Tensor", "Vector", "Matrix", "Graph",
    "Pipeline", "Transformer", "Encoder", "Decoder", "Attention",
    "Cache", "Store", "Repository", "Connection", "Session",
    "Request", "Response", "Handler", "Middleware", "Router",
    "Scheduler", "Worker", "Executor", "Future", "Promise",
    "Iterator", "Stream", "Buffer", "Channel", "Queue",
    "Config", "Settings", "Options", "Builder", "Factory",
    "Tokenizer", "Embedding", "Layer", "Block", "Module",
    "Optimizer", "Loss", "Metric", "Trainer", "Evaluator",
]

_FUNC_NAMES = [
    "compute", "transform", "process", "analyze", "extract",
    "aggregate", "filter", "map", "reduce", "merge",
    "validate", "sanitize", "normalize", "serialize", "deserialize",
    "encode", "decode", "compress", "decompress", "hash",
    "search", "find", "match", "replace", "split",
    "train", "predict", "evaluate", "optimize", "update",
    "connect", "disconnect", "send", "receive", "broadcast",
    "subscribe", "publish", "listen", "emit", "dispatch",
    "create", "delete", "update", "read", "list_all",
]

_TYPE_NAMES = [
    "int", "float", "str", "bool", "bytes",
    "List[int]", "List[str]", "Dict[str, Any]", "Optional[int]",
    "Optional[str]", "Tuple[int, int]", "Sequence[float]",
    "Union[int, float]", "Union[str, bytes]",
    "Tensor", "ndarray", "DataFrame", "Graph",
    "Callable[[int], str]", "Iterator[int]",
]


def _pick(items: list[str], n: int = 1) -> list[str]:
    return random.sample(items, min(n, len(items)))


def gen_module_doc() -> str:
    """Generate a module-level docstring with overview."""
    prefix = random.choice(_MODULE_PREFIXES)
    purposes = [
        "High-performance data processing library",
        "Neural network building blocks",
        "Distributed computing primitives",
        "Streaming data pipeline framework",
        "Memory-efficient tensor operations",
        "Asynchronous task scheduling system",
        "Type-safe configuration management",
        "Event-driven architecture components",
        "Machine learning model utilities",
        "Graph computation framework",
    ]
    purpose = random.choice(purposes)
    return f'"""\n{prefix}\n{"=" * len(prefix)}\n\n{purpose}.\n\nFeatures:\n- Efficient memory management\n- Thread-safe operations\n- Comprehensive error handling\n- Streaming support\n"""\n'


def gen_class_doc() -> str:
    """Generate a class definition with docstring and methods."""
    cls_name = random.choice(_CLASS_NAMES)
    n_attrs = random.randint(2, 6)
    n_methods = random.randint(2, 5)

    lines = [f"class {cls_name}:"]
    purposes = [
        f"Manages {cls_name.lower()} operations with automatic batching",
        f"Provides thread-safe access to {cls_name.lower()} data",
        f"Implements {cls_name.lower()} transformation pipeline",
        f"Handles {cls_name.lower()} lifecycle and resource management",
    ]
    lines.append(f'    """{random.choice(purposes)}."""')
    lines.append("")

    # Class attributes
    for attr in _pick([f"_{c}" for c in "abcdefghij"], n_attrs):
        attr_type = random.choice(_TYPE_NAMES)
        lines.append(f"    {attr}: {attr_type}")
    lines.append("")

    # __init__
    params = _pick([f"_{c}" for c in "abcdefghij"], min(n_attrs, 3))
    typed_params = []
    for p in params:
        ptype = random.choice(_TYPE_NAMES)
        default = random.choice(["None", "0", "''", "True", "False", "[]", "{}"])
        typed_params.append(f"{p}: {ptype} = {default}")
    lines.append(f"    def __init__(self, {', '.join(typed_params)}):")
    for p in params:
        lines.append(f"        self.{p} = {p}")
    lines.append("")

    # Methods
    for _ in range(n_methods):
        fn_name = random.choice(_FUNC_NAMES)
        n_params = random.randint(0, 4)
        fn_params = _pick(["x", "y", "z", "data", "key", "value", "idx", "n", "fn", "callback"], n_params)
        ret_type = random.choice(["int", "float", "str", "bool", "None", "List[int]", "Optional[str]"])

        typed_fn_params = []
        for p in fn_params:
            pt = random.choice(_TYPE_NAMES)
            typed_fn_params.append(f"{p}: {pt}")

        lines.append(f"    def {fn_name}(self, {', '.join(typed_fn_params)}) -> {ret_type}:")
        lines.append(f'        """{fn_name} operation on {cls_name.lower()} instance."""')

        # Implementation hint
        impl_patterns = [
            f"        result = self._compute({', '.join(fn_params) if fn_params else ''})",
            f"        return self._{fn_name}_impl({', '.join(fn_params) if fn_params else ''})",
            f"        self._validate_inputs({', '.join(fn_params) if fn_params else ''})",
            f"        return self._process({', '.join(fn_params) if fn_params else ''})",
        ]
        lines.append(random.choice(impl_patterns))
        lines.append("")

    return "\n".join(lines)


def gen_function_doc() -> str:
    """Generate a standalone function with full type annotations and docstring."""
    fn_name = random.choice(_FUNC_NAMES)
    n_params = random.randint(1, 5)
    params = _pick(["x", "y", "z", "data", "key", "value", "idx", "n", "threshold", "weight", "callback", "items"], n_params)
    ret_type = random.choice(["int", "float", "str", "bool", "None", "List[int]", "Dict[str, Any]", "Tuple[int, int]", "Optional[float]"])

    typed_params = []
    for p in params:
        pt = random.choice(_TYPE_NAMES)
        default = random.choice(["", " = None", " = 0", " = ''", " = True", " = []"])
        typed_params.append(f"{p}: {pt}{default}")

    lines = [f"def {fn_name}({', '.join(typed_params)}) -> {ret_type}:"]
    lines.append(f'    """{fn_name.capitalize()} with proper validation."""')
    lines.append("")

    # Parameter descriptions in docstring
    for p in params:
        lines.append(f"    Args:")
        lines.append(f"        {p}: Input parameter for {fn_name}")
    lines.append(f"    Returns:")
    lines.append(f"        {ret_type}: Computed result")
    lines.append("")

    # Example usage
    args = [str(random.randint(1, 20)) for _ in params]
    lines.append(f"    Example:")
    lines.append(f"        >>> {fn_name}({', '.join(args)})")

    return "\n".join(lines)


def gen_usage_example() -> str:
    """Generate usage examples combining classes and functions."""
    cls_name = random.choice(_CLASS_NAMES)
    fn_names = _pick(_FUNC_NAMES, 2)
    lines = [f"# Usage example for {cls_name}"]
    lines.append(f"obj = {cls_name}()")

    for fn in fn_names:
        args = [str(random.randint(1, 10)) for _ in range(random.randint(0, 3))]
        lines.append(f"result = obj.{fn}({', '.join(args)})")
        lines.append(f"print(f'{{result}}')  # Output depends on inputs")

    lines.append(f"# Error handling")
    lines.append(f"try:")
    lines.append(f"    obj.{fn_names[0]}(-1)  # invalid input")
    lines.append(f"except ValueError as e:")
    lines.append(f"    print(f'Error: {{e}}')")

    return "\n".join(lines)


def gen_mixed_doc(seed: int | None = None) -> str:
    """Generate a random documentation snippet."""
    if seed is not None:
        random.seed(seed)

    generators = [gen_module_doc, gen_class_doc, gen_function_doc, gen_usage_example]
    weights = [0.15, 0.40, 0.30, 0.15]
    chosen = random.choices(generators, weights=weights, k=1)[0]
    return chosen()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pillar 2: API docs")
    parser.add_argument("--output", default="pillar2.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} API doc samples...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = gen_mixed_doc()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples}", file=sys.stderr)

    print(f"Done: {args.samples} samples -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
