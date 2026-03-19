"""gen_pillar4_grpo.py — Self-verification training data (v2).

v2: Generates attempt->fail->reason->fix cycles instead of simple QA pairs.
Each sample teaches the model to verify its own output, identify errors,
reason about corrections, and produce fixed code.

Format: <task>...<attempt_1>...<execute>...<reasoning>...<attempt_2>...<execute>...<done/>

Output: JSONL with "text" field (no reward field — used as LM training data).

Usage:
    python gen_pillar4_grpo.py --output pillar4.jsonl --samples 50000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


_MATH_TASKS = [
    {
        "task": "Compute the factorial of 7.",
        "wrong": "7! = 7 * 6 * 5 * 4 * 3 = 2520",
        "execution_wrong": "Error: 2520 is incorrect. Missing the * 2 * 1 part.",
        "reasoning": "I forgot to multiply by 2 and 1. Factorial means 7*6*5*4*3*2*1.",
        "fixed": "7! = 7 * 6 * 5 * 4 * 3 * 2 * 1 = 5040",
    },
    {
        "task": "What is the GCD of 48 and 36?",
        "wrong": "GCD(48, 36) = 6",
        "execution_wrong": "Error: 6 divides both but is not the greatest.",
        "reasoning": "Let me check: 48 = 2^4 * 3, 36 = 2^2 * 3^2. Common: 2^2 * 3 = 12.",
        "fixed": "GCD(48, 36) = 12. Both 48/12=4 and 36/12=3 are integers.",
    },
    {
        "task": "What is the sum of the first 100 natural numbers?",
        "wrong": "Sum = 100 * 99 / 2 = 4950",
        "execution_wrong": "Error: Should use n*(n+1)/2, not n*(n-1)/2.",
        "reasoning": "Formula is n*(n+1)/2 = 100*101/2. I used 99 instead of 101.",
        "fixed": "Sum = 100 * 101 / 2 = 5050",
    },
    {
        "task": "What is 15% of 200?",
        "wrong": "15% of 200 = 200 * 0.15 = 25",
        "execution_wrong": "Error: 200 * 0.15 = 30, not 25.",
        "reasoning": "I calculated 200 * 0.125 instead of 200 * 0.15.",
        "fixed": "15% of 200 = 200 * 0.15 = 30",
    },
    {
        "task": "Convert 1001 from decimal to binary.",
        "wrong": "1001 in binary = 111101001",
        "execution_wrong": "Error: 111101001 in binary = 489, not 1001.",
        "reasoning": "Let me recompute: 1001/2=500r1, 500/2=250r0, 250/2=125r0, 125/2=62r1, 62/2=31r0, 31/2=15r1, 15/2=7r1, 7/2=3r1, 3/2=1r1, 1/2=0r1.",
        "fixed": "1001 in binary = 1111101001",
    },
]


_CODE_TASKS = [
    {
        "task": "Write a Python function that reverses a list in-place.",
        "wrong": (
            "def reverse_list(lst):\n"
            "    return lst.reverse()"
        ),
        "execution_wrong": "Error: list.reverse() returns None, not the list.",
        "reasoning": "reverse() modifies in-place and returns None. Should return the list after calling reverse().",
        "fixed": (
            "def reverse_list(lst):\n"
            "    lst.reverse()\n"
            "    return lst"
        ),
    },
    {
        "task": "Write a function that checks if a string is a palindrome.",
        "wrong": (
            "def is_palindrome(s):\n"
            "    return s == s.reverse()"
        ),
        "execution_wrong": "Error: strings don't have a reverse() method. AttributeError.",
        "reasoning": "Need to use slicing [::-1] instead of .reverse() which is a list method.",
        "fixed": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]"
        ),
    },
    {
        "task": "Write a function to find the maximum element in a list.",
        "wrong": (
            "def find_max(lst):\n"
            "    max_val = lst[0]\n"
            "    for i in lst:\n"
            "        if i > max_val:\n"
            "            max_val = i\n"
            "    return max_val"
        ),
        "execution_wrong": "Error: IndexError when lst is empty — lst[0] fails.",
        "reasoning": "Need to handle the edge case of an empty list before accessing lst[0].",
        "fixed": (
            "def find_max(lst):\n"
            "    if not lst:\n"
            "        return None\n"
            "    max_val = lst[0]\n"
            "    for i in lst:\n"
            "        if i > max_val:\n"
            "            max_val = i\n"
            "    return max_val"
        ),
    },
    {
        "task": "Write a function that counts vowels in a string.",
        "wrong": (
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for ch in s:\n"
            "        if ch in 'aeiou':\n"
            "            count += 1\n"
            "    return count"
        ),
        "execution_wrong": "Error: count_vowels('HELLO') returns 1 instead of 2. Case sensitivity.",
        "reasoning": "Need to convert to lowercase before checking, or include uppercase vowels.",
        "fixed": (
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for ch in s.lower():\n"
            "        if ch in 'aeiou':\n"
            "            count += 1\n"
            "    return count"
        ),
    },
    {
        "task": "Write a function that removes duplicates from a list.",
        "wrong": (
            "def remove_dupes(lst):\n"
            "    return list(set(lst))"
        ),
        "execution_wrong": "Error: remove_dupes([3,1,2,3]) returns [1,2,3] but order is not preserved.",
        "reasoning": "set() doesn't preserve order. Should use dict.fromkeys() to preserve insertion order.",
        "fixed": (
            "def remove_dupes(lst):\n"
            "    return list(dict.fromkeys(lst))"
        ),
    },
]


def _gen_arithmetic_task() -> dict:
    """Generate a procedural arithmetic self-verification task."""
    a = random.randint(2, 100)
    b = random.randint(2, 100)
    op = random.choice(["+", "-", "*"])
    correct = eval(f"{a} {op} {b}")
    offset = random.choice([-2, -1, 1, 2, 5, -5, 10, -10])
    wrong = correct + offset

    return {
        "task": f"What is {a} {op} {b}?",
        "wrong": f"{a} {op} {b} = {wrong}",
        "execution_wrong": f"Error: {a} {op} {b} = {correct}, not {wrong}.",
        "reasoning": f"Let me recalculate: {a} {op} {b} = {correct}. I made an arithmetic error.",
        "fixed": f"{a} {op} {b} = {correct}",
    }


def _gen_code_task() -> dict:
    """Generate a procedural code self-verification task."""
    tasks = [
        {
            "task": "Write a function that returns the average of a list of numbers.",
            "gen_wrong": lambda: (
                "def average(nums):\n"
                "    return sum(nums) / len(nums)"
            ),
            "exec_wrong": "Error: ZeroDivisionError when nums is empty.",
            "reasoning": "Need to check for empty list before dividing.",
            "gen_fixed": lambda: (
                "def average(nums):\n"
                "    if not nums:\n"
                "        return 0\n"
                "    return sum(nums) / len(nums)"
            ),
        },
        {
            "task": "Write a function that flattens a nested list (one level deep).",
            "gen_wrong": lambda: (
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        result.append(item)\n"
                "    return result"
            ),
            "exec_wrong": "Error: flatten([[1,2],[3,4]]) returns [[1,2],[3,4]] instead of [1,2,3,4].",
            "reasoning": "Need to extend instead of append to flatten one level.",
            "gen_fixed": lambda: (
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        result.extend(item)\n"
                "    return result"
            ),
        },
    ]
    return random.choice(tasks)


def gen_self_verification_sample() -> str:
    """Generate a single self-verification sample."""
    # 50% template-based, 50% procedural
    if random.random() < 0.5:
        if random.random() < 0.5:
            task_data = random.choice(_MATH_TASKS)
        else:
            task_data = random.choice(_CODE_TASKS)
    else:
        if random.random() < 0.5:
            task_data = _gen_arithmetic_task()
        else:
            task_data = _gen_code_task()

    parts = [
        f"<task>",
        task_data["task"],
        f"</task>",
        f"",
        f"<attempt_1>",
        task_data["wrong"],
        f"</attempt_1>",
        f"",
        f"<execute>",
        task_data["execution_wrong"],
        f"</execute>",
        f"",
        f"<reasoning>",
        task_data["reasoning"],
        f"</reasoning>",
        f"",
        f"<attempt_2>",
        task_data["fixed"],
        f"</attempt_2>",
        f"",
        f"<execute>",
        "Success.",
        f"</execute>",
        f"",
        f"<done/>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pillar 4: self-verification data")
    parser.add_argument("--output", default="pillar4.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} self-verification samples...",
          file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = gen_self_verification_sample()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples}", file=sys.stderr)

    print(f"Done: {args.samples} samples -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
