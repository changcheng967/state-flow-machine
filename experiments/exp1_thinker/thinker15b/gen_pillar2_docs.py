"""gen_pillar2_docs.py — Enhanced synthetic documentation generator (v2).

v2 adds 5 stages:
  A. API documentation (original)
  B. Bug injection + fix (code with deliberate bugs and corrections)
  C. Few-shot examples (problem + multiple solutions with explanations)
  D. Multi-file context (imports, dependencies, cross-file references)
  E. Cross-language (Python + equivalent in JavaScript/Rust/Go)

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
]

_CLASS_NAMES = [
    "DataFrame", "Tensor", "Vector", "Matrix", "Graph",
    "Pipeline", "Encoder", "Decoder", "Attention",
    "Cache", "Store", "Repository", "Connection", "Session",
    "Request", "Response", "Handler", "Middleware", "Router",
    "Scheduler", "Worker", "Executor", "Future",
    "Iterator", "Stream", "Buffer", "Channel", "Queue",
    "Config", "Settings", "Options", "Builder", "Factory",
]

_FUNC_NAMES = [
    "compute", "transform", "process", "analyze", "extract",
    "aggregate", "filter", "map", "reduce", "merge",
    "validate", "sanitize", "normalize", "serialize", "deserialize",
    "encode", "decode", "compress", "decompress", "hash",
    "search", "find", "match", "replace", "split",
    "train", "predict", "evaluate", "optimize", "update",
    "connect", "disconnect", "send", "receive", "broadcast",
    "create", "delete", "read", "list_all",
]

_TYPE_NAMES = [
    "int", "float", "str", "bool", "bytes",
    "List[int]", "List[str]", "Dict[str, Any]", "Optional[int]",
    "Optional[str]", "Tuple[int, int]", "Sequence[float]",
    "Union[int, float]", "Union[str, bytes]",
    "Tensor", "ndarray", "DataFrame", "Graph",
]


def _pick(items: list, n: int = 1) -> list:
    return random.sample(items, min(n, len(items)))


# ── Stage A: API documentation (original) ────────────────────────────

def gen_module_doc() -> str:
    prefix = random.choice(_MODULE_PREFIXES)
    purposes = [
        "High-performance data processing library",
        "Neural network building blocks",
        "Distributed computing primitives",
        "Streaming data pipeline framework",
        "Event-driven architecture components",
    ]
    purpose = random.choice(purposes)
    return (f'"""\n{prefix}\n{"=" * len(prefix)}\n\n'
            f'{purpose}.\n\nFeatures:\n'
            f'- Efficient memory management\n'
            f'- Thread-safe operations\n"""\n')


def gen_class_doc() -> str:
    cls_name = random.choice(_CLASS_NAMES)
    n_methods = random.randint(2, 5)
    lines = [f"class {cls_name}:"]
    purposes = [
        f"Manages {cls_name.lower()} operations",
        f"Thread-safe {cls_name.lower()} handler",
        f"Implements {cls_name.lower()} pipeline",
    ]
    lines.append(f'    """{random.choice(purposes)}."""\n')

    params = _pick([f"_{c}" for c in "abcdefghij"], random.randint(2, 4))
    typed_params = [f"{p}: {random.choice(_TYPE_NAMES)} = None" for p in params]
    lines.append(f"    def __init__(self, {', '.join(typed_params)}):")
    for p in params:
        lines.append(f"        self.{p} = {p}")
    lines.append("")

    for _ in range(n_methods):
        fn = random.choice(_FUNC_NAMES)
        fn_params = _pick(["x", "y", "z", "data", "key", "idx"], random.randint(0, 3))
        ret = random.choice(["int", "float", "str", "bool", "None", "List[int]"])
        typed_fn = [f"{p}: {random.choice(_TYPE_NAMES)}" for p in fn_params]
        lines.append(f"    def {fn}(self, {', '.join(typed_fn)}) -> {ret}:")
        lines.append(f'        """{fn} operation."""')
        lines.append(f"        return self._{fn}_impl()")
        lines.append("")

    return "\n".join(lines)


def gen_function_doc() -> str:
    fn_name = random.choice(_FUNC_NAMES)
    params = _pick(["x", "y", "z", "data", "key", "idx", "n"], random.randint(1, 4))
    ret = random.choice(["int", "float", "str", "bool", "None", "List[int]"])
    typed = [f"{p}: {random.choice(_TYPE_NAMES)}" for p in params]
    lines = [f"def {fn_name}({', '.join(typed)}) -> {ret}:"]
    lines.append(f'    """{fn_name.capitalize()} with proper validation."""')
    for p in params:
        lines.append(f"    # {p}: input parameter")
    lines.append(f"    pass")
    return "\n".join(lines)


# ── Stage B: Bug injection + fix ────────────────────────────────────

def gen_bug_fix() -> str:
    bugs = [
        {
            "title": "Off-by-one error in range",
            "wrong": "for i in range(1, len(items)):\n    process(items[i])",
            "fix": "for i in range(len(items)):\n    process(items[i])",
            "explain": "Bug: range(1, n) skips index 0. Fix: range(n) iterates 0..n-1.",
        },
        {
            "title": "Mutable default argument",
            "wrong": "def add_item(item, lst=[]):\n    lst.append(item)\n    return lst",
            "fix": "def add_item(item, lst=None):\n    if lst is None:\n        lst = []\n    lst.append(item)\n    return lst",
            "explain": "Bug: default [] is shared across calls. Fix: use None and create new list.",
        },
        {
            "title": "Integer division vs float division",
            "wrong": "average = total / count  # Python 2 style",
            "fix": "average = total / count if isinstance(total, float) else total // count",
            "explain": "Bug: / is float division in Python 3. If int division needed, use //.",
        },
        {
            "title": "Missing return in recursion",
            "wrong": (
                "def binary_search(arr, target, lo, hi):\n"
                "    if lo > hi:\n"
                "        return -1\n"
                "    mid = (lo + hi) // 2\n"
                "    if arr[mid] == target:\n"
                "        return mid\n"
                "    binary_search(arr, target, lo, mid - 1)"
            ),
            "fix": (
                "def binary_search(arr, target, lo, hi):\n"
                "    if lo > hi:\n"
                "        return -1\n"
                "    mid = (lo + hi) // 2\n"
                "    if arr[mid] == target:\n"
                "        return mid\n"
                "    return binary_search(arr, target, lo, mid - 1)"
            ),
            "explain": "Bug: missing return on recursive call. Always returns None for left half.",
        },
        {
            "title": "String comparison with is",
            "wrong": 'if user_input is "admin":\n    grant_access()',
            "fix": 'if user_input == "admin":\n    grant_access()',
            "explain": "Bug: 'is' checks identity, not equality. Use == for value comparison.",
        },
        {
            "title": "Modifying list while iterating",
            "wrong": (
                "for item in items:\n"
                "    if should_remove(item):\n"
                "        items.remove(item)"
            ),
            "fix": (
                "items = [item for item in items if not should_remove(item)]"
            ),
            "explain": "Bug: removing from list while iterating skips elements. Fix: list comprehension.",
        },
    ]

    bug = random.choice(bugs)
    return (
        f"# Bug Fix: {bug['title']}\n\n"
        f"## Buggy Code:\n```\n{bug['wrong']}\n```\n\n"
        f"## Fixed Code:\n```\n{bug['fix']}\n```\n\n"
        f"## Explanation:\n{bug['explain']}\n"
    )


# ── Stage C: Few-shot examples ──────────────────────────────────────

def gen_few_shot() -> str:
    problems = [
        {
            "problem": "Reverse a string without using built-in reverse.",
            "solution_1": (
                "def reverse(s):\n"
                "    return s[::-1]"
            ),
            "solution_2": (
                "def reverse(s):\n"
                "    result = ''\n"
                "    for ch in s:\n"
                "        result = ch + result\n"
                "    return result"
            ),
            "explanation": "Solution 1 uses slicing (O(n) time, O(n) space). "
                          "Solution 2 builds string by prepending (O(n^2) time due to "
                          "string immutability, but works without slicing).",
        },
        {
            "problem": "Find the two numbers in a list that add up to a target.",
            "solution_1": (
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return [seen[complement], i]\n"
                "        seen[num] = i"
            ),
            "solution_2": (
                "def two_sum(nums, target):\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(i+1, len(nums)):\n"
                "            if nums[i] + nums[j] == target:\n"
                "                return [i, j]"
            ),
            "explanation": "Solution 1 uses a hash map (O(n) time, O(n) space). "
                          "Solution 2 uses brute force (O(n^2) time, O(1) space).",
        },
        {
            "problem": "Check if a number is prime.",
            "solution_1": (
                "def is_prime(n):\n"
                "    if n < 2: return False\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if n % i == 0: return False\n"
                "    return True"
            ),
            "solution_2": (
                "def is_prime(n):\n"
                "    if n < 2: return False\n"
                "    if n == 2: return True\n"
                "    if n % 2 == 0: return False\n"
                "    for i in range(3, int(n**0.5) + 1, 2):\n"
                "        if n % i == 0: return False\n"
                "    return True"
            ),
            "explanation": "Solution 1 checks all divisors up to sqrt(n). "
                          "Solution 2 skips even numbers after checking 2 (2x faster).",
        },
    ]

    prob = random.choice(problems)
    return (
        f"## Problem: {prob['problem']}\n\n"
        f"### Solution 1:\n````\n{prob['solution_1']}\n```\n\n"
        f"### Solution 2:\n```\n{prob['solution_2']}\n```\n\n"
        f"### Analysis:\n{prob['explanation']}\n"
    )


# ── Stage D: Multi-file context ─────────────────────────────────────

def gen_multi_file() -> str:
    module = random.choice(_MODULE_PREFIXES)
    cls = random.choice(_CLASS_NAMES)

    file_a = (
        f"# {module}/models.py\n"
        f"class {cls}:\n"
        f"    def __init__(self, name: str):\n"
        f"        self.name = name\n"
        f"        self._data = []\n\n"
        f"    def add(self, item):\n"
        f"        self._data.append(item)\n\n"
        f"    def get_all(self):\n"
        f"        return list(self._data)"
    )

    file_b = (
        f"# {module}/handlers.py\n"
        f"from .models import {cls}\n\n"
        f"class {cls}Handler:\n"
        f"    def __init__(self, name: str):\n"
        f"        self.model = {cls}(name)\n\n"
        f"    def process(self, items: list) -> int:\n"
        f"        for item in items:\n"
        f"            self.model.add(item)\n"
        f"        return len(self.model.get_all())"
    )

    file_c = (
        f"# {module}/__init__.py\n"
        f"from .models import {cls}\n"
        f"from .handlers import {cls}Handler\n\n"
        f"__all__ = ['{cls}', '{cls}Handler']"
    )

    usage = (
        f"# Usage example\n"
        f"from {module} import {cls}Handler\n\n"
        f"handler = {cls}Handler('test')\n"
        f"count = handler.process([1, 2, 3])\n"
        f"print(f'Processed {{count}} items')"
    )

    return (
        f"## File: {module}/models.py\n```\n{file_a}\n```\n\n"
        f"## File: {module}/handlers.py\n```\n{file_b}\n```\n\n"
        f"## File: {module}/__init__.py\n```\n{file_c}\n```\n\n"
        f"## Usage:\n```\n{usage}\n```\n"
    )


# ── Stage E: Cross-language ─────────────────────────────────────────

def gen_cross_language() -> str:
    tasks = [
        {
            "description": "Fibonacci sequence (iterative)",
            "python": (
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b"
            ),
            "javascript": (
                "function fibonacci(n) {\n"
                "    if (n <= 1) return n;\n"
                "    let a = 0, b = 1;\n"
                "    for (let i = 2; i <= n; i++) {\n"
                "        [a, b] = [b, a + b];\n"
                "    }\n"
                "    return b;\n"
                "}"
            ),
            "rust": (
                "fn fibonacci(n: u32) -> u64 {\n"
                "    if n <= 1 { return n as u64; }\n"
                "    let mut a: u64 = 0;\n"
                "    let mut b: u64 = 1;\n"
                "    for _ in 2..=n {\n"
                "        let temp = a + b;\n"
                "        a = b;\n"
                "        b = temp;\n"
                "    }\n"
                "    b\n"
                "}"
            ),
        },
        {
            "description": "Binary search",
            "python": (
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1"
            ),
            "javascript": (
                "function binarySearch(arr, target) {\n"
                "    let lo = 0, hi = arr.length - 1;\n"
                "    while (lo <= hi) {\n"
                "        const mid = Math.floor((lo + hi) / 2);\n"
                "        if (arr[mid] === target) return mid;\n"
                "        if (arr[mid] < target) lo = mid + 1;\n"
                "        else hi = mid - 1;\n"
                "    }\n"
                "    return -1;\n"
                "}"
            ),
            "go": (
                "func binarySearch(arr []int, target int) int {\n"
                "    lo, hi := 0, len(arr)-1\n"
                "    for lo <= hi {\n"
                "        mid := (lo + hi) / 2\n"
                "        if arr[mid] == target { return mid }\n"
                "        if arr[mid] < target { lo = mid + 1 } else { hi = mid - 1 }\n"
                "    }\n"
                "    return -1\n"
                "}"
            ),
        },
    ]

    task = random.choice(tasks)
    langs = ["python", "javascript"]
    if random.random() < 0.5:
        langs.append("rust")
    else:
        langs.append("go")

    parts = [f"## {task['description']}\n"]
    for lang in langs:
        code = task[lang]
        parts.append(f"### {lang.capitalize()}:\n```\n{code}\n```\n")

    return "\n".join(parts)


# ── Main generation ─────────────────────────────────────────────────

_GENERATORS = {
    "A": [gen_module_doc, gen_class_doc, gen_function_doc],
    "B": [gen_bug_fix],
    "C": [gen_few_shot],
    "D": [gen_multi_file],
    "E": [gen_cross_language],
}

_STAGE_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]


def gen_mixed_doc(seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)
    stage = random.choices(
        ["A", "B", "C", "D", "E"], weights=_STAGE_WEIGHTS, k=1)[0]
    gen = random.choice(_GENERATORS[stage])
    return gen()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pillar 2: enhanced docs (v2)")
    parser.add_argument("--output", default="pillar2.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} enhanced doc samples...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = gen_mixed_doc()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples}", file=sys.stderr)

    print(f"Done: {args.samples} samples -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
