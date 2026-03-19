"""gen_pillar4_grpo.py — GRPO question-answer pairs with binary rewards.

Generates groups of 4-8 answers per question, each with a binary reward
(1.0 = correct, 0.0 = incorrect). Used for Group Relative Policy
Optimization (GRPO) training.

Output: JSONL with "text" and "reward" fields.

Usage:
    python gen_pillar4_grpo.py --output pillar4.jsonl --samples 50000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


_MATH_QUESTIONS = [
    ("What is 17 * 23?", ["391", "17 * 23 = 391", "The product is 391"], ["392", "393", "390", "400", "17 * 23 = 392"]),
    ("What is the square root of 144?", ["12", "sqrt(144) = 12", "The square root is 12"], ["14", "11", "13", "10"]),
    ("What is 2^10?", ["1024", "2 to the power of 10 is 1024"], ["512", "2048", "256", "1028"]),
    ("What is 15% of 200?", ["30", "15% of 200 = 30"], ["25", "35", "20", "40"]),
    ("What is the GCD of 48 and 36?", ["12", "GCD(48, 36) = 12"], ["6", "18", "24", "8"]),
    ("What is 7! (7 factorial)?", ["5040", "7! = 5040"], ["720", "40320", "504", "50400"]),
    ("What is 1001 in binary?", ["1111101001", "1001 in binary is 1111101001"], ["111101001", "11111001", "1100100001"]),
    ("What is the sum of the first 100 natural numbers?", ["5050", "Sum = 100*101/2 = 5050"], ["5000", "5100", "505", "4950"]),
    ("What is log2(1024)?", ["10", "log base 2 of 1024 = 10"], ["11", "8", "20", "100"]),
    ("What is the LCM of 12 and 18?", ["36", "LCM(12, 18) = 36"], ["24", "72", "48", "18"]),
]

_CODE_QUESTIONS = [
    ("What does list(range(5)) produce?", ["[0, 1, 2, 3, 4]", "It produces [0, 1, 2, 3, 4]"], ["[1, 2, 3, 4, 5]", "[0, 1, 2, 3, 4, 5]", "[1, 2, 3, 4]"]),
    ("What is the output of: 'hello'[1:4]?", ["'ell'", "It outputs 'ell'"], ["'hel'", "'ell '", "'ello'"]),
    ("What does len({'a': 1, 'b': 2}) return?", ["2", "It returns 2"], ["1", "3", "4", "0"]),
    ("What is the type of True in Python?", ["bool", "bool (subclass of int)", "It is of type bool"], ["int", "str", "NoneType", "TrueType"]),
    ("What does [x**2 for x in range(4)] produce?", ["[0, 1, 4, 9]", "It produces [0, 1, 4, 9]"], ["[1, 4, 9, 16]", "[0, 2, 4, 6]", "[0, 1, 2, 3]"]),
    ("What does ' '.join(['a', 'b', 'c']) return?", ["'a b c'", "It returns 'a b c'"], ["'abc'", "'a,b,c'", "'a  b  c'"]),
    ("What is the result of 10 // 3?", ["3", "Integer division gives 3"], ["3.33", "4", "3.0", "0"]),
    ("What does sorted([3, 1, 4, 1, 5]) return?", ["[1, 1, 3, 4, 5]", "[1, 1, 3, 4, 5]"], ["[3, 1, 4, 1, 5]", "[5, 4, 3, 1, 1]"]),
]

_REASONING_QUESTIONS = [
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
     ["No, we cannot conclude that", "This is a logical fallacy - some flowers fading doesn't mean some roses do"],
     ["Yes, since roses are flowers", "No, but all roses must fade"]),
    ("A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     ["$0.05", "5 cents", "The ball costs $0.05"],
     ["$0.10", "$0.15", "10 cents", "$1.00"]),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
     ["5 minutes", "Each machine makes 1 widget in 5 minutes"],
     ["100 minutes", "20 minutes", "1 minute", "500 minutes"]),
    ("How many times do the hour and minute hands of a clock overlap in 12 hours?",
     ["11", "They overlap 11 times in 12 hours"],
     ["12", "10", "24", "22"]),
    ("If you have 6 identical socks in a drawer (3 black, 3 white), what is the minimum number you must draw to guarantee a matching pair?",
     ["2", "With 6 socks total, drawing 2 guarantees a pair by pigeonhole"],
     ["3", "4", "6", "1"]),
]

_ALL_QUESTIONS = _MATH_QUESTIONS + _CODE_QUESTIONS + _REASONING_QUESTIONS


def _gen_answer(question: str, correct: bool) -> str:
    """Generate an answer for a question, correct or incorrect."""
    # Find the matching question in our bank
    for q, correct_answers, wrong_answers in _ALL_QUESTIONS:
        if q == question:
            if correct:
                answer = random.choice(correct_answers)
            else:
                answer = random.choice(wrong_answers)
            return f"Question: {question}\nAnswer: {answer}"

    # Fallback for unknown questions
    return f"Question: {question}\nAnswer: The answer is {random.randint(1, 100)}."


def gen_grpo_group() -> list[dict]:
    """Generate one GRPO group (4-8 answers per question)."""
    question, _, _ = random.choice(_ALL_QUESTIONS)
    group_size = random.randint(4, 8)

    # At least one correct, at least one incorrect
    n_correct = random.randint(1, max(1, group_size // 2))
    n_incorrect = group_size - n_correct

    samples = []
    for _ in range(n_correct):
        text = _gen_answer(question, correct=True)
        samples.append({"text": text, "reward": 1.0})

    for _ in range(n_incorrect):
        text = _gen_answer(question, correct=False)
        samples.append({"text": text, "reward": 0.0})

    random.shuffle(samples)
    return samples


def gen_custom_question() -> list[dict]:
    """Generate a GRPO group with a procedurally-created question."""
    q_type = random.choice(["arithmetic", "logic", "code"])

    if q_type == "arithmetic":
        a, b = random.randint(2, 50), random.randint(2, 50)
        op = random.choice(["+", "-", "*"])
        question = f"What is {a} {op} {b}?"
        correct = str(eval(f"{a} {op} {b}"))
        wrong_options = [
            str(int(correct) + random.choice([-2, -1, 1, 2, 5, 10, -5, -10])),
            str(int(correct) + random.choice([-3, 3, 7, -7])),
        ]
    elif q_type == "logic":
        n = random.randint(1, 10)
        question = f"What is the {n}th prime number?"
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        correct = str(primes[n - 1])
        wrong_options = [str(p) for p in primes if p != primes[n - 1]][:3]
    else:
        var = random.choice(["x", "y", "z"])
        val = random.randint(1, 100)
        inc = random.randint(1, 10)
        question = f"What is the value of {var} after: {var} = {val}; {var} += {inc}?"
        correct = str(val + inc)
        wrong_options = [str(val), str(val - 1), str(val + 1)]

    group_size = random.randint(4, 6)
    n_correct = random.randint(1, max(1, group_size // 2))
    n_incorrect = group_size - n_correct

    samples = []
    for _ in range(n_correct):
        text = f"Question: {question}\nAnswer: {correct}"
        samples.append({"text": text, "reward": 1.0})

    for i in range(n_incorrect):
        wrong = wrong_options[i % len(wrong_options)] if wrong_options else "0"
        text = f"Question: {question}\nAnswer: {wrong}"
        samples.append({"text": text, "reward": 0.0})

    random.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pillar 4: GRPO pairs")
    parser.add_argument("--output", default="pillar4.jsonl", help="Output JSONL file")
    parser.add_argument("--groups", type=int, default=25000, help="Number of question groups")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.groups} GRPO groups...", file=sys.stderr)

    total_samples = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.groups):
            if random.random() < 0.5:
                samples = gen_grpo_group()
            else:
                samples = gen_custom_question()
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                total_samples += 1
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{args.groups} groups ({total_samples} samples)", file=sys.stderr)

    print(f"Done: {args.groups} groups ({total_samples} samples) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
