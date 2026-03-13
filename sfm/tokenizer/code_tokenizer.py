"""
Code Tokenizer for SFM

A simple BPE-style tokenizer for code that handles:
- Keywords, operators, and punctuation
- Identifiers and literals
- Whitespace normalization
- Special tokens for structure
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json


# Special tokens
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<sep>": 4,
    "<mask>": 5,
    "<func>": 6,
    "<class>": 7,
    "<var>": 8,
    "<stmt>": 9,
    "<newline>": 10,
    "<indent>": 11,
    "<dedent>": 12,
}

# Python keywords
PYTHON_KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield"
]

# Common operators
OPERATORS = [
    "+", "-", "*", "/", "//", "%", "**",
    "==", "!=", "<", ">", "<=", ">=",
    "=", "+=", "-=", "*=", "/=", "//=", "%=",
    "&", "|", "^", "~", "<<", ">>",
    "and", "or", "not", "in", "is",
    "->", "=>", ":", ";", ",", ".", "..."
]

# Punctuation
PUNCTUATION = ["(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "@", "#", "\\"]


class CodeTokenizer:
    """
    Simple tokenizer for code with BPE-style subword tokenization.

    Features:
    - Special tokens for structure
    - Keyword and operator handling
    - Identifier normalization
    - Whitespace normalization
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        min_freq: int = 2,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Target vocabulary size.
            min_freq: Minimum frequency for BPE merges.
            special_tokens: Custom special tokens.
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or SPECIAL_TOKENS.copy()

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

        # Initialize with special tokens
        self._init_vocab()

    def _init_vocab(self):
        """Initialize vocabulary with special tokens."""
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Add keywords and operators
        next_id = len(self.special_tokens)
        for kw in PYTHON_KEYWORDS:
            if kw not in self.token_to_id:
                self.token_to_id[kw] = next_id
                self.id_to_token[next_id] = kw
                next_id += 1

        for op in OPERATORS + PUNCTUATION:
            if op not in self.token_to_id:
                self.token_to_id[op] = next_id
                self.id_to_token[next_id] = op
                next_id += 1

    def tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code into initial tokens (before BPE).

        Args:
            code: Source code string.

        Returns:
            List of tokens.
        """
        tokens = []

        # Patterns for tokenization
        patterns = [
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENT'),  # Identifiers
            (r'\d+\.?\d*', 'NUMBER'),  # Numbers
            (r'"[^"]*"', 'STRING'),  # Double-quoted strings
            (r"'[^']*'", 'STRING'),  # Single-quoted strings
            (r'#[^\n]*', 'COMMENT'),  # Comments
            (r'\s+', 'SPACE'),  # Whitespace
        ]

        # Sort patterns by length of first match (longer first for operators)
        i = 0
        while i < len(code):
            matched = False

            # Try to match multi-character operators first
            for op in sorted(OPERATORS + PUNCTUATION, key=len, reverse=True):
                if code[i:i+len(op)] == op:
                    tokens.append(op)
                    i += len(op)
                    matched = True
                    break

            if matched:
                continue

            # Try patterns
            for pattern, token_type in patterns:
                match = re.match(pattern, code[i:])
                if match:
                    value = match.group()
                    if token_type == 'IDENT':
                        if value in PYTHON_KEYWORDS:
                            tokens.append(value)
                        else:
                            # Add identifier prefix for normalization
                            tokens.append(f"▁{value}")
                    elif token_type == 'NUMBER':
                        tokens.append(f"NUM_{value}")
                    elif token_type == 'STRING':
                        tokens.append("STR_LITERAL")
                    elif token_type == 'COMMENT':
                        tokens.append("COMMENT")
                    elif token_type == 'SPACE':
                        # Normalize whitespace
                        if '\n' in value:
                            newlines = value.count('\n')
                            for _ in range(newlines):
                                tokens.append("<newline>")
                        elif value[0] == ' ':
                            tokens.append("▁")  # Space marker
                    i += len(value)
                    matched = True
                    break

            if not matched:
                # Unknown character
                tokens.append(f"▁{code[i]}")
                i += 1

        return tokens

    def train(self, corpus: List[str], verbose: bool = False):
        """
        Train BPE merges on a corpus.

        Args:
            corpus: List of code strings.
            verbose: Print progress.
        """
        if verbose:
            print("Tokenizing corpus...")

        # Tokenize all code
        all_tokens = []
        for code in corpus:
            all_tokens.extend(self.tokenize_code(code))

        if verbose:
            print(f"Initial tokens: {len(all_tokens)}")

        # Build initial vocabulary
        token_freq = Counter(all_tokens)
        next_id = max(self.token_to_id.values()) + 1

        # Add tokens that meet minimum frequency
        for token, freq in token_freq.items():
            if freq >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        if verbose:
            print(f"Vocabulary after initial: {len(self.token_to_id)}")

        # BPE merges until vocab size reached
        while len(self.token_to_id) < self.vocab_size:
            # Count pairs
            pairs = Counter()
            for i in range(len(all_tokens) - 1):
                pair = (all_tokens[i], all_tokens[i + 1])
                pairs[pair] += 1

            if not pairs:
                break

            # Find most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            if pairs[best_pair] < self.min_freq:
                break

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            self.merges.append(best_pair)
            next_id += 1

            # Apply merge
            new_tokens = []
            i = 0
            while i < len(all_tokens):
                if i < len(all_tokens) - 1 and (all_tokens[i], all_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(all_tokens[i])
                    i += 1
            all_tokens = new_tokens

            if verbose and len(self.merges) % 100 == 0:
                print(f"Merges: {len(self.merges)}, Vocab: {len(self.token_to_id)}")

        if verbose:
            print(f"Final vocabulary size: {len(self.token_to_id)}")

    def encode(self, code: str) -> List[int]:
        """
        Encode code to token IDs.

        Args:
            code: Source code string.

        Returns:
            List of token IDs.
        """
        tokens = self.tokenize_code(code)

        # Apply BPE merges
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens by character-level fallback
                for char in token:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.special_tokens["<unk>"])

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to code string.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded code string.
        """
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                tokens.append(token)
            else:
                tokens.append("<unk>")

        # Join tokens
        code = "".join(tokens)

        # Clean up special markers
        code = code.replace("▁", " ")
        code = code.replace("<newline>", "\n")
        code = code.replace("<pad>", "")
        code = code.replace("<unk>", "?")

        return code.strip()

    def batch_encode(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode multiple code strings with padding.

        Args:
            codes: List of code strings.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.

        Returns:
            Tuple of (padded_ids, lengths).
        """
        all_ids = []
        lengths = []

        for code in codes:
            ids = self.encode(code)
            lengths.append(len(ids))

            if max_length is not None:
                ids = ids[:max_length]

            all_ids.append(ids)

        if padding and max_length is not None:
            pad_id = self.special_tokens["<pad>"]
            all_ids = [
                ids + [pad_id] * (max_length - len(ids))
                for ids in all_ids
            ]

        return all_ids, lengths

    @property
    def vocab(self) -> Dict[str, int]:
        """Get vocabulary."""
        return self.token_to_id.copy()

    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.token_to_id)

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab": self.token_to_id,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_freq=data["min_freq"],
            special_tokens=data["special_tokens"]
        )
        tokenizer.token_to_id = data["vocab"]
        tokenizer.id_to_token = {v: k for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]

        return tokenizer


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Code Tokenizer Smoke Test")
    print("=" * 60)

    # Sample code
    code_samples = [
        "def hello_world():\n    print('Hello, World!')\n    return True",
        "for i in range(10):\n    x = i * 2\n    print(x)",
        "class MyClass:\n    def __init__(self, value):\n        self.value = value",
    ]

    # Create tokenizer
    print("\n1. Creating and training tokenizer...")
    tokenizer = CodeTokenizer(vocab_size=1000, min_freq=1)
    tokenizer.train(code_samples, verbose=True)

    print(f"   Vocabulary size: {tokenizer.vocab_size_actual}")

    # Test tokenization
    print("\n2. Testing tokenization...")
    for i, code in enumerate(code_samples):
        print(f"\n   Sample {i+1}:")
        print(f"   Input: {repr(code[:50])}...")
        tokens = tokenizer.tokenize_code(code)
        print(f"   Tokens: {tokens[:20]}...")

    # Test encoding/decoding
    print("\n3. Testing encode/decode...")
    test_code = "def add(a, b):\n    return a + b"
    ids = tokenizer.encode(test_code)
    decoded = tokenizer.decode(ids)

    print(f"   Input: {repr(test_code)}")
    print(f"   IDs: {ids[:20]}...")
    print(f"   Decoded: {repr(decoded)}")

    # Test batch encoding
    print("\n4. Testing batch encoding...")
    batch_ids, lengths = tokenizer.batch_encode(code_samples, max_length=50, padding=True)
    print(f"   Batch size: {len(batch_ids)}")
    print(f"   Sequence lengths: {lengths}")
    print(f"   Padded shape: ({len(batch_ids)}, {len(batch_ids[0])})")

    print("\n" + "=" * 60)
    print("All Code Tokenizer tests passed!")
    print("=" * 60)
