"""gen_pillar3_trajectories.py — Tool-call trajectories (ReAct style).

Generates ReAct-style Thought/Action/Observation/Answer cycles with
multi-tool chains and chain-of-thought reasoning.

Output: JSONL with "text" field.

Usage:
    python gen_pillar3_trajectories.py --output pillar3.jsonl --samples 300000
"""

from __future__ import annotations

import argparse
import json
import random
import sys


# ── Tool definitions: (action_template, obs_factory) ────────────────
# Templates use $placeholder syntax (NOT {placeholder}) to avoid
# clashing with Python .format() or f-string braces.

_TOOL_ACTIONS: dict[str, list[tuple[str, object]]] = {
    "calculator": [
        ("add($a, $b)", lambda: f"Result: {random.randint(1, 1000)}"),
        ("multiply($a, $b)", lambda: f"Result: {random.randint(1, 10000)}"),
        ("subtract($a, $b)", lambda: f"Result: {random.randint(-100, 100)}"),
        ("divide($a, $b)", lambda: f"Result: {round(random.uniform(0.1, 100), 2)}"),
    ],
    "search": [
        ("search($query)", lambda: "Found 5 relevant results: ..."),
        ("lookup($term)", lambda: f"Definition: {random.choice(['a data structure', 'an algorithm', 'a design pattern', 'a mathematical concept', 'a programming paradigm'])}"),
        ("find_similar($item)", lambda: f"Similar items: {', '.join(random.sample(['alpha', 'beta', 'gamma', 'delta', 'epsilon'], 3))}"),
    ],
    "database": [
        ("query('SELECT * FROM users WHERE active=true')", lambda: f"Returned {random.randint(1, 100)} rows"),
        ("get_by_id($id)", lambda: "Record: {id: 1, name: 'example', value: 42}"),
        ("count($table)", lambda: f"Count: {random.randint(10, 500)}"),
        ("filter($table, $condition)", lambda: f"Filtered {random.randint(1, 50)} records"),
    ],
    "file_system": [
        ("read_file($path)", lambda: f"Contents: {random.choice(['data loaded successfully', 'file contains 150 lines', 'JSON parsed: 10 objects'])}"),
        ("list_dir($path)", lambda: f"Files: {', '.join(random.sample(['main.py', 'config.json', 'data.csv', 'README.md', 'utils.py'], 3))}"),
        ("write_file($path, $content)", lambda: "File written successfully"),
        ("parse_json($path)", lambda: "Parsed: {key: value, items: [...]}"),
    ],
    "api_client": [
        ("get($url)", lambda: f"Response 200: {random.choice(['status: ok', 'data: [...]', 'count: ' + str(random.randint(1, 100))])}"),
        ("post($url, $body)", lambda: "Response 201: Created successfully"),
        ("put($url, $body)", lambda: "Response 200: Updated successfully"),
        ("delete($url)", lambda: "Response 204: Deleted successfully"),
    ],
    "text_analysis": [
        ("summarize($text)", lambda: f"Summary: {random.choice(['The text discusses key concepts', 'Main points extracted successfully', 'Core argument summarized'])}"),
        ("extract_entities($text)", lambda: f"Entities: [{', '.join(random.sample(['PERSON', 'ORG', 'DATE', 'LOCATION', 'PRODUCT'], 3))}]"),
        ("classify($text)", lambda: f"Category: {random.choice(['technical', 'business', 'creative', 'academic', 'casual'])}"),
        ("translate($text, $target_lang)", lambda: f"Translation: [translated text in {random.choice(['French', 'German', 'Spanish', 'Chinese', 'Japanese'])}]"),
    ],
}


def _fill_action(template: str) -> str:
    """Replace $placeholders in an action template with random values."""
    replacements = {
        "$a": str(random.randint(1, 100)),
        "$b": str(random.randint(1, 100)),
        "$query": f"'{random.choice(['python', 'algorithm', 'design pattern', 'data structure'])}'",
        "$term": f"'{random.choice(['recursion', 'polymorphism', 'caching', 'hashing'])}'",
        "$item": f"'{random.choice(['item1', 'item2', 'example'])}'",
        "$id": str(random.randint(1, 1000)),
        "$table": f"'{random.choice(['users', 'products', 'orders'])}'",
        "$condition": f"'{random.choice(['active = true', 'age > 18', 'status = pending'])}'",
        "$path": f"'{random.choice(['/data/file.csv', '/config/settings.json'])}'",
        "$content": f"'{random.choice(['new data', 'updated config'])}'",
        "$url": f"'/api/{random.choice(['users', 'items', 'data'])}'",
        "$body": "'{key: value}'",
        "$text": "'sample text'",
        "$target_lang": f"'{random.choice(['french', 'spanish', 'german'])}'",
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


_TOPICS = [
    "Calculate the total cost of items in a shopping cart",
    "Find and compare information about programming languages",
    "Process and analyze a dataset of user records",
    "Debug a file parsing issue in a data pipeline",
    "Integrate multiple API endpoints for a dashboard",
    "Summarize and classify a collection of documents",
    "Design a database schema for an e-commerce system",
    "Build a text processing pipeline for sentiment analysis",
    "Optimize a search query across multiple data sources",
    "Create a file processing workflow for batch operations",
    "Analyze performance metrics from log files",
    "Implement a data validation pipeline",
    "Compare results from different computational methods",
    "Set up automated data extraction from multiple sources",
    "Design a caching strategy for API responses",
]


def gen_single_step_trajectory() -> str:
    """Generate a 1-2 tool call trajectory."""
    lines = []
    lines.append(f"Question: {random.choice(_TOPICS)}")
    lines.append("")

    tool_name = random.choice(list(_TOOL_ACTIONS.keys()))
    action_template, obs_fn = random.choice(_TOOL_ACTIONS[tool_name])
    action = _fill_action(action_template)

    lines.append(f"Thought: I need to {random.choice(['calculate', 'find', 'query', 'read', 'analyze', 'process'])} this. I'll use the {tool_name} tool.")
    lines.append(f"Action: {tool_name}")
    lines.append(f"Action Input: {action}")
    lines.append(f"Observation: {obs_fn()}")
    lines.append("")
    lines.append(f"Answer: Based on the {tool_name} result, I can conclude the answer.")

    return "\n".join(lines)


def gen_multi_step_trajectory(num_steps: int = 3) -> str:
    """Generate a multi-tool chain trajectory with reasoning."""
    lines = []
    topic = random.choice(_TOPICS)
    lines.append(f"Question: {topic}")
    lines.append("")

    tools_used = random.sample(list(_TOOL_ACTIONS.keys()),
                               min(num_steps, len(_TOOL_ACTIONS)))

    for step, tool_name in enumerate(tools_used):
        action_template, obs_fn = random.choice(_TOOL_ACTIONS[tool_name])
        action = _fill_action(action_template)

        if step == 0:
            thought = f"I need to start by using {tool_name} to get initial data."
        else:
            thought = (f"From the previous step, I now need to "
                       f"{random.choice(['refine', 'combine', 'validate', 'transform'])} "
                       f"the results using {tool_name}.")

        lines.append(f"Thought: {thought}")
        lines.append(f"Action: {tool_name}")
        lines.append(f"Action Input: {action}")
        lines.append(f"Observation: {obs_fn()}")
        lines.append("")

    lines.append("Thought: I now have all the information needed to answer.")
    lines.append(f"Answer: After {len(tools_used)} steps of analysis, the answer is clear. "
                 f"Using {', '.join(tools_used)}, I determined the result by combining "
                 f"information from each tool.")

    return "\n".join(lines)


def gen_trajectory(seed: int | None = None) -> str:
    """Generate a random trajectory (single or multi-step)."""
    if seed is not None:
        random.seed(seed)

    num_steps = random.choices(
        [1, 2, 3, 4, 5], weights=[0.25, 0.30, 0.25, 0.15, 0.05])[0]
    if num_steps <= 2:
        return gen_single_step_trajectory()
    return gen_multi_step_trajectory(num_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pillar 3: trajectories")
    parser.add_argument("--output", default="pillar3.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
