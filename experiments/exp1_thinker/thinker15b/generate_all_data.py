"""generate_all_data.py — Run all 4 generators, train BPE tokenizer, tokenize.

v2 changes:
  - Pillar 1 uses exec()-based traces (no reward field)
  - Pillar 2 has 5 stages (A-E) including bug fixes, few-shot, multi-file
  - Pillar 3 has multi-turn sessions with [TURN] separator
  - Pillar 4 is self-verification data (no reward field)
  - Tokenizer reserves EOS and [TURN] special tokens

Pipeline:
1. Generate pillar1-4 data
2. Train ByteLevelBPE tokenizer on merged corpus
3. Tokenize into fixed-length chunks (seq_len=4096)
4. Save vocab.json, merges.json, train_chunks.npy

Usage:
    python generate_all_data.py --output_dir ./data --seq_len 4096 --vocab_size 32000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ByteLevelBPE


def run_generators(output_dir: str, pillar1_n: int = 150000,
                   pillar2_n: int = 200000,
                   pillar3_n: int = 150000,
                   pillar4_n: int = 50000) -> tuple:
    """Run all 4 generators and collect text samples."""
    print("=" * 60, file=sys.stderr)
    print("STEP 1: Generating synthetic data (v2)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_texts: list[str] = []

    # Pillar 1: Execution traces (exec-based)
    p1_path = os.path.join(output_dir, "pillar1.jsonl")
    print(f"\n[Pillar 1] Execution traces: {pillar1_n} samples", file=sys.stderr)
    t0 = time.time()
    _run_cmd(f'python gen_pillar1_traces.py --output "{p1_path}" '
             f'--samples {pillar1_n}')
    p1_texts = _read_jsonl_texts(p1_path)
    all_texts.extend(p1_texts)
    print(f"  Generated {len(p1_texts)} texts in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # Pillar 2: Enhanced docs (5 stages)
    p2_path = os.path.join(output_dir, "pillar2.jsonl")
    print(f"\n[Pillar 2] Enhanced docs: {pillar2_n} samples", file=sys.stderr)
    t0 = time.time()
    _run_cmd(f'python gen_pillar2_docs.py --output "{p2_path}" '
             f'--samples {pillar2_n}')
    p2_texts = _read_jsonl_texts(p2_path)
    all_texts.extend(p2_texts)
    print(f"  Generated {len(p2_texts)} texts in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # Pillar 3: Trajectories (with multi-turn)
    p3_path = os.path.join(output_dir, "pillar3.jsonl")
    print(f"\n[Pillar 3] Trajectories: {pillar3_n} samples", file=sys.stderr)
    t0 = time.time()
    _run_cmd(f'python gen_pillar3_trajectories.py --output "{p3_path}" '
             f'--samples {pillar3_n}')
    p3_texts = _read_jsonl_texts(p3_path)
    all_texts.extend(p3_texts)
    print(f"  Generated {len(p3_texts)} texts in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # Pillar 4: Self-verification (v2, no reward field)
    p4_path = os.path.join(output_dir, "pillar4.jsonl")
    print(f"\n[Pillar 4] Self-verification: {pillar4_n} samples", file=sys.stderr)
    t0 = time.time()
    _run_cmd(f'python gen_pillar4_grpo.py --output "{p4_path}" '
             f'--samples {pillar4_n}')
    p4_texts = _read_jsonl_texts(p4_path)
    all_texts.extend(p4_texts)
    print(f"  Generated {len(p4_texts)} texts in {time.time()-t0:.1f}s",
          file=sys.stderr)

    print(f"\nTotal texts collected: {len(all_texts):,}", file=sys.stderr)
    return all_texts


def _run_cmd(cmd: str) -> None:
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Command failed: {cmd}", file=sys.stderr)
        print(f"  stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")


def _read_jsonl_texts(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    texts.append(text)
    return texts


def train_tokenizer(texts: list[str], vocab_size: int,
                    output_dir: str) -> ByteLevelBPE:
    print("\n" + "=" * 60, file=sys.stderr)
    print("STEP 2: Training BPE tokenizer (v2)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    max_train = 2_000_000
    if len(texts) > max_train:
        import random
        random.seed(42)
        train_texts = random.sample(texts, max_train)
    else:
        train_texts = texts

    print(f"Training on {len(train_texts):,} texts, target vocab={vocab_size}",
          file=sys.stderr)
    t0 = time.time()
    tok = ByteLevelBPE()
    tok.train(train_texts, vocab_size=vocab_size, min_freq=2)
    print(f"Tokenizer trained in {time.time()-t0:.1f}s: "
          f"{len(tok.encoder)} vocab, {len(tok.merges)} merges, "
          f"EOS={tok.eos_id}, [TURN]={tok.turn_id}", file=sys.stderr)

    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.json")
    tok.save(vocab_path, merges_path)
    print(f"Saved: {vocab_path}, {merges_path}", file=sys.stderr)

    return tok


def tokenize_texts(tok: ByteLevelBPE, texts: list[str],
                   seq_len: int, output_dir: str) -> None:
    print("\n" + "=" * 60, file=sys.stderr)
    print("STEP 3: Tokenizing corpus", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    import numpy as np

    print(f"Tokenizing {len(texts):,} texts into chunks of {seq_len}...",
          file=sys.stderr)
    t0 = time.time()

    all_ids: list[int] = []
    eos = tok.eos_id
    for i, text in enumerate(texts):
        ids = tok.encode(text)
        all_ids.extend(ids)
        all_ids.append(eos)
        if (i + 1) % 50000 == 0:
            print(f"  {i+1}/{len(texts)} docs, {len(all_ids):,} tokens",
                  file=sys.stderr)

    total_tokens = len(all_ids)
    usable = (total_tokens // seq_len) * seq_len
    print(f"Total tokens: {total_tokens:,}, usable: {usable:,} "
          f"({usable // seq_len:,} chunks)", file=sys.stderr)

    trimmed = all_ids[:usable]
    chunks = np.array(trimmed, dtype=np.int32).reshape(-1, seq_len)

    np.random.seed(42)
    np.random.shuffle(chunks)

    out_path = os.path.join(output_dir, "train_chunks.npy")
    np.save(out_path, chunks)
    print(f"Saved {chunks.shape} chunks to {out_path}", file=sys.stderr)
    print(f"Tokenization done in {time.time()-t0:.1f}s", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all v2 data and train tokenizer")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--pillar1", type=int, default=150000)
    parser.add_argument("--pillar2", type=int, default=200000)
    parser.add_argument("--pillar3", type=int, default=150000)
    parser.add_argument("--pillar4", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}", file=sys.stderr)

    all_texts = run_generators(
        args.output_dir,
        pillar1_n=args.pillar1,
        pillar2_n=args.pillar2,
        pillar3_n=args.pillar3,
        pillar4_n=args.pillar4,
    )

    tok = train_tokenizer(all_texts, args.vocab_size, args.output_dir)
    tokenize_texts(tok, all_texts, args.seq_len, args.output_dir)

    print("\n" + "=" * 60, file=sys.stderr)
    print("DATA GENERATION COMPLETE (v2)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Files in {args.output_dir}:", file=sys.stderr)
    for f in sorted(os.listdir(args.output_dir)):
        fp = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
