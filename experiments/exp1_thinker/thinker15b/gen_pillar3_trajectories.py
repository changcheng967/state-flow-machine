"""gen_pillar3_trajectories.py — Tool-call trajectories with multi-turn (v2).

v2: Adds multi-turn sessions with [TURN] separator between conversation turns.
Each session can have 1-3 turns, building on previous context.

Output: JSONL with "text" field.

Usage:
    python gen_pillar3_trajectories.py --output pillar3.jsonl --samples 300000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


_TOOL_ACTIONS: dict[str, list[tuple[str, object]]] = {
    "calculator": [
        ("add($a, $b)", lambda: f"Result: {random.randint(1, 1000)}"),
        ("multiply($a, $b)", lambda: f"Result: {random.randint(1, 10000)}"),
        ("subtract($a, $b)", lambda: f"Result: {random.randint(-100, 100)}"),
        ("divide($a, $b)", lambda: f"Result: {round(random.uniform(0.1, 100), 2)}"),
    ],
    "search": [
        ("search($query)", lambda: "Found 5 relevant results: ..."),
        ("lookup($term)", lambda: f"Definition: {random.choice(['a data structure', 'an algorithm', 'a design pattern', 'a mathematical concept'])}"),
        ("find_similar($item)", lambda: f"Similar items: {', '.join(random.sample(['alpha', 'beta', 'gamma', 'delta'], 3))}"),
    ],
    "database": [
        ("query('SELECT * FROM users WHERE active=true')", lambda: f"Returned {random.randint(1, 100)} rows"),
        ("get_by_id($id)", lambda: "Record: {id: 1, name: 'example', value: 42}"),
        ("count($table)", lambda: f"Count: {random.randint(10, 500)}"),
    ],
    "file_system": [
        ("read_file($path)", lambda: f"Contents: {random.choice(['data loaded', 'file contains 150 lines', 'JSON parsed'])}"),
        ("list_dir($path)", lambda: f"Files: {', '.join(random.sample(['main.py', 'config.json', 'data.csv'], 3))}"),
        ("write_file($path, $content)", lambda: "File written successfully"),
    ],
    "api_client": [
        ("get($url)", lambda: f"Response 200: {random.choice(['status: ok', 'data: [...]'])}"),
        ("post($url, $body)", lambda: "Response 201: Created"),
    ],
    "text_analysis": [
        ("summarize($text)", lambda: f"Summary: {random.choice(['Key concepts extracted', 'Main points summarized'])}"),
        ("extract_entities($text)", lambda: f"Entities: [{', '.join(random.sample(['PERSON', 'ORG', 'DATE'], 3))}]"),
        ("classify($text)", lambda: f"Category: {random.choice(['technical', 'business', 'creative'])}"),
    ],
}


def _fill_action(template: str) -> str:
    replacements = {
        "$a": str(random.randint(1, 100)),
        "$b": str(random.randint(1, 100)),
        "$query": f"'{random.choice(['python', 'algorithm', 'design pattern'])}'",
        "$term": f"'{random.choice(['recursion', 'polymorphism', 'caching'])}'",
        "$item": f"'{random.choice(['item1', 'item2', 'example'])}'",
        "$id": str(random.randint(1, 1000)),
        "$table": f"'{random.choice(['users', 'products', 'orders'])}'",
        "$path": f"'{random.choice(['/data/file.csv', '/config/settings.json'])}'",
        "$content": f"'{random.choice(['new data', 'updated config'])}'",
        "$url": f"'/api/{random.choice(['users', 'items', 'data'])}'",
        "$body": "'{key: value}'",
        "$text": "'sample text'",
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


_TOPICS = [
    "Calculate the total cost of items in a shopping cart",
    "Find and compare programming language features",
    "Process a dataset of user records",
    "Debug a file parsing issue in a data pipeline",
    "Integrate multiple API endpoints for a dashboard",
    "Summarize a collection of documents",
    "Design a database schema for an e-commerce system",
    "Build a text processing pipeline",
    "Optimize a search query across data sources",
    "Create a file processing workflow",
    "Analyze performance metrics from log files",
    "Implement a data validation pipeline",
]


def gen_single_turn() -> str:
    """Generate a single-turn trajectory."""
    lines = []
    topic = random.choice(_TOPICS)
    lines.append(f"Question: {topic}")
    lines.append("")

    num_steps = random.choices([1, 2, 3, 4], weights=[0.25, 0.35, 0.25, 0.15])[0]
    tools_used = random.sample(list(_TOOL_ACTIONS.keys()),
                               min(num_steps, len(_TOOL_ACTIONS)))

    for step, tool_name in enumerate(tools_used):
        action_template, obs_fn = random.choice(_TOOL_ACTIONS[tool_name])
        action = _fill_action(action_template)

        if step == 0:
            thought = f"I need to use {tool_name} to get the information."
        else:
            thought = (f"From the previous step, I now need to "
                       f"{random.choice(['refine', 'combine', 'validate'])} "
                       f"using {tool_name}.")

        lines.append(f"Thought: {thought}")
        lines.append(f"Action: {tool_name}")
        lines.append(f"Action Input: {action}")
        lines.append(f"Observation: {obs_fn()}")
        lines.append("")

    lines.append("Thought: I now have all the information needed.")
    lines.append(f"Answer: Using {', '.join(tools_used)}, the result is clear.")

    return "\n".join(lines)


def gen_multi_turn_session() -> str:
    """Generate a multi-turn session with [TURN] separators."""
    turns: list[str] = []
    num_turns = random.choices([2, 3], weights=[0.6, 0.4])[0]

    # Turn 1: initial question
    topic = random.choice(_TOPICS)
    turn1 = [f"Question: {topic}", ""]

    tool_name = random.choice(list(_TOOL_ACTIONS.keys()))
    action_template, obs_fn = random.choice(_TOOL_ACTIONS[tool_name])
    action = _fill_action(action_template)
    turn1.append(f"Thought: I'll use {tool_name} to start.")
    turn1.append(f"Action: {tool_name}")
    turn1.append(f"Action Input: {action}")
    turn1.append(f"Observation: {obs_fn()}")
    turn1.append("")
    turn1.append(f"Answer: Initial result from {tool_name}.")
    turns.append("\n".join(turn1))

    # Follow-up turns
    followups = [
        "Can you get more details on that?",
        "What if I change the parameters?",
        "Can you also check the related data?",
        "How does this compare to the previous result?",
        "Can you summarize all findings so far?",
    ]

    for t in range(1, num_turns):
        followup = followups[t % len(followups)]
        turn = [f"Follow-up: {followup}", ""]

        tool2 = random.choice(list(_TOOL_ACTIONS.keys()))
        action2_template, obs2_fn = random.choice(_TOOL_ACTIONS[tool2])
        action2 = _fill_action(action2_template)

        turn.append(f"Thought: Following up with {tool2}.")
        turn.append(f"Action: {tool2}")
        turn.append(f"Action Input: {action2}")
        turn.append(f"Observation: {obs2_fn()}")
        turn.append("")
        turn.append(f"Answer: Updated result combining previous findings.")

        turns.append("\n".join(turn))

    return "\n[TURN]\n".join(turns)


def gen_trajectory(seed: int | None = None) -> str:
    """Generate a random trajectory (single or multi-turn)."""
    if seed is not None:
        random.seed(seed)

    if random.random() < 0.4:
        return gen_multi_turn_session()
    return gen_single_turn()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pillar 3: trajectories (v2 with multi-turn)")
    parser.add_argument("--output", default="pillar3.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.samples} trajectory samples...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.samples):
            text = gen_trajectory()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{args.samples}", file=sys.stderr)

    print(f"Done: {args.samples} samples -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
