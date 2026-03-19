"""tokenizer.py — Byte-level BPE tokenizer, stdlib only.

Implements a GPT-2 style byte-level BPE tokenizer from scratch.
No external dependencies (no tokenizers, no transformers).

v2 changes: [TURN] separator token reserved at slot_vocab_size boundary
to allow multi-turn training with explicit turn boundaries.

Usage:
    tok = ByteLevelBPE()
    tok.train(texts, vocab_size=32000)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    tok.save("vocab.json", "merges.json")

    # Round-trip test:
    python tokenizer.py --test-roundtrip
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
)


class ByteLevelBPE:
    """Byte-level BPE tokenizer.

    Handles all UTF-8 text via byte fallback. Unknown bytes map
    through GPT-2's byte-to-unicode table.

    Special tokens:
      - EOS: reserved at vocab boundary
      - [TURN]: separator for multi-turn conversations
    """

    def __init__(self) -> None:
        self.b2u: dict[int, str] = _bytes_to_unicode()
        self.u2b: dict[str, int] = {v: k for k, v in self.b2u.items()}
        self.encoder: dict[str, int] = {}
        self.decoder: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        self.eos_id: int = 0
        self.turn_id: int = 0

    def train(self, corpus_texts: list[str], vocab_size: int = 32000,
              min_freq: int = 2) -> None:
        """Train BPE on a list of text strings.

        Args:
            corpus_texts: training corpus (list of documents).
            vocab_size: target vocabulary size (256 byte tokens + merges + special).
            min_freq: minimum pair frequency to merge.
        """
        word_freqs: Counter[str] = Counter()
        for text in corpus_texts:
            for word in _GPT2_PAT.findall(text):
                byte_chars = "".join(
                    self.b2u[b] for b in word.encode("utf-8"))
                word_freqs[byte_chars] += 1

        word_splits: dict[tuple[str, ...], int] = {}
        for word_str, freq in word_freqs.items():
            word_splits[tuple(word_str)] = freq

        self.encoder = {}
        for byte_val in range(256):
            ch = self.b2u[byte_val]
            self.encoder[ch] = len(self.encoder)

        self.merges = []
        num_merges = vocab_size - 256 - 2  # reserve EOS + [TURN]
        for _ in range(num_merges):
            pairs: Counter[tuple[str, str]] = Counter()
            for word_tuple, freq in word_splits.items():
                for i in range(len(word_tuple) - 1):
                    pairs[(word_tuple[i], word_tuple[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)  # type: ignore[arg-type]
            if pairs[best_pair] < min_freq:
                break

            new_word_splits: dict[tuple[str, ...], int] = {}
            merged_token = best_pair[0] + best_pair[1]
            for word_tuple, freq in word_splits.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if (i < len(word_tuple) - 1
                            and word_tuple[i] == best_pair[0]
                            and word_tuple[i + 1] == best_pair[1]):
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_splits[tuple(new_word)] = freq
            word_splits = new_word_splits

            self.merges.append(best_pair)
            self.encoder[merged_token] = 256 + len(self.merges) - 1

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Special tokens: EOS and [TURN]
        self.eos_id = len(self.encoder)
        self.encoder["<EOS>"] = self.eos_id
        self.decoder[self.eos_id] = "<EOS>"

        self.turn_id = len(self.encoder)
        self.encoder["[TURN]"] = self.turn_id
        self.decoder[self.turn_id] = "[TURN]"

    def _bpe(self, token: str) -> list[str]:
        """Apply BPE merges to a single pre-tokenized word."""
        word = list(token)
        if len(word) <= 1:
            return word

        while True:
            best_pair: tuple[str, str] | None = None
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_pair is None:
                break

            word = (word[:best_idx]
                    + [best_pair[0] + best_pair[1]]
                    + word[best_idx + 2:])
            if len(word) == 1:
                break

        return word

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        token_ids: list[int] = []
        for word in _GPT2_PAT.findall(text):
            byte_chars = "".join(
                self.b2u[b] for b in word.encode("utf-8"))
            bpe_tokens = self._bpe(byte_chars)

            for tok in bpe_tokens:
                tid = self.encoder.get(tok)
                if tid is not None:
                    token_ids.append(tid)
                else:
                    for ch in tok:
                        byte_val = self.u2b.get(ch)
                        if byte_val is not None and byte_val < 256:
                            token_ids.append(self.encoder.get(
                                self.b2u[byte_val], 0))
                        else:
                            token_ids.append(0)

        return token_ids

    def encode_with_turns(self, turns: list[str]) -> list[int]:
        """Encode multiple turns separated by [TURN] tokens.

        Args:
            turns: list of text strings, one per turn.
        """
        token_ids: list[int] = []
        for i, turn in enumerate(turns):
            if i > 0:
                token_ids.append(self.turn_id)
            token_ids.extend(self.encode(turn))
        token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        chars: list[str] = []
        for tid in token_ids:
            tok_str = self.decoder.get(tid, "")
            if tok_str in ("<EOS>", "[TURN]"):
                continue
            for ch in tok_str:
                byte_val = self.u2b.get(ch)
                if byte_val is not None and byte_val < 256:
                    chars.append(chr(byte_val))
        return "".join(chars)

    def save(self, vocab_path: str, merges_path: str) -> None:
        """Save vocabulary and merges to JSON files."""
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump([" ".join(pair) for pair in self.merges], f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str, merges_path: str) -> ByteLevelBPE:
        """Load vocabulary and merges from JSON files."""
        tok = cls()
        with open(vocab_path, "r", encoding="utf-8") as f:
            tok.encoder = json.load(f)
        tok.decoder = {int(v): k for k, v in tok.encoder.items()}
        with open(merges_path, "r", encoding="utf-8") as f:
            merge_strings = json.load(f)
        tok.merges = [tuple(m.split(" ", 1)) for m in merge_strings]
        tok.bpe_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.eos_id = tok.encoder.get("<EOS>", len(tok.encoder))
        tok.turn_id = tok.encoder.get("[TURN]", len(tok.encoder))
        return tok


def _test_roundtrip() -> None:
    """Quick encode/decode roundtrip test."""
    tok = ByteLevelBPE()
    samples = [
        "hello world",
        "def foo(x):\n    return x + 1\n",
        "x = x + y * z  # arithmetic",
        "if x > 0 and y < 100:\n    print(x)\n",
        'print("hello")',
        "",
    ]
    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        status = "OK" if decoded == text else "FAIL"
        if status == "FAIL":
            print(f"  [{status}] {text!r} -> {ids} -> {decoded!r}")
        else:
            print(f"  [{status}] len={len(ids)}: {text!r}")

    # Test with trained tokenizer
    print("\nTraining on small corpus...")
    corpus = [
        "def add(a, b):\n    return a + b\n",
        "def mul(a, b):\n    return a * b\n",
        "x = 10\ny = 20\nz = x + y\n",
        "for i in range(10):\n    print(i)\n",
        "class Foo:\n    def __init__(self, x):\n        self.x = x\n",
    ] * 100

    tok2 = ByteLevelBPE()
    tok2.train(corpus, vocab_size=500)
    print(f"  Vocab size: {len(tok2.encoder)}")
    print(f"  Merges: {len(tok2.merges)}")
    print(f"  EOS id: {tok2.eos_id}")
    print(f"  [TURN] id: {tok2.turn_id}")

    for text in ["def add(a, b):", "x = 10 + 20", "print(i)"]:
        ids = tok2.encode(text)
        decoded = tok2.decode(ids)
        status = "OK" if decoded == text else "FAIL"
        print(f"  [{status}] {text!r} -> {ids[:20]}... -> {decoded!r}")

    # Test multi-turn encoding
    turns = ["What is 2+2?", "2+2 = 4", "Now what about 3*7?"]
    multi_ids = tok2.encode_with_turns(turns)
    decoded_multi = tok2.decode(multi_ids)
    print(f"  Multi-turn encode: {len(multi_ids)} tokens")
    print(f"  [TURN] token found: {tok2.turn_id in multi_ids}")

    # Save/load roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        vp = os.path.join(tmpdir, "vocab.json")
        mp = os.path.join(tmpdir, "merges.json")
        tok2.save(vp, mp)
        tok3 = ByteLevelBPE.load(vp, mp)
        for text in ["x = 10", "def foo():", "hello"]:
            ids1 = tok2.encode(text)
            ids2 = tok3.encode(text)
            assert ids1 == ids2, f"Mismatch: {ids1} vs {ids2}"
        assert tok2.turn_id == tok3.turn_id
        print("  Save/load roundtrip: OK")


if __name__ == "__main__":
    if "--test-roundtrip" in sys.argv:
        print("ByteLevelBPE roundtrip test")
        _test_roundtrip()
    else:
        print("Usage: python tokenizer.py --test-roundtrip")
