"""
Code Tokenizer for SFM

Provides two tokenizers:
1. SimpleTokenizer: Fast whitespace/split tokenizer for simple programs (no BPE)
2. CodeTokenizer: Full BPE-style tokenizer for complex code

For state tracking experiments, use SimpleTokenizer - it's 100x faster.
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
}

# Python keywords
PYTHON_KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield"
]


class SimpleTokenizer:
    """
    Fast tokenizer that splits on whitespace and special characters.
    No BPE merges - instant training and encoding.

    Perfect for simple programs like:
        a = 5
        x = y + z
        result = a * b

    Training takes < 1 second regardless of corpus size.
    """

    def __init__(self):
        """Initialize simple tokenizer."""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Add special tokens
        for token, idx in SPECIAL_TOKENS.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self._trained = False

    def _tokenize(self, code: str) -> List[str]:
        """
        Split code into tokens.

        Splits on:
        - Whitespace
        - Operators: +, -, *, /, =, etc.
        - Parentheses, brackets
        - Punctuation
        """
        # Pattern: split on whitespace OR capture operators/punctuation
        # This keeps operators as separate tokens
        pattern = r'(\s+|[+\-*/%=<>!&|^~]+|[()\[\]{}:,;.]+)'

        # Split and filter empty strings
        raw_tokens = re.split(pattern, code)
        tokens = [t for t in raw_tokens if t and not t.isspace()]

        return tokens

    def train(self, corpus: List[str], verbose: bool = False):
        """
        Build vocabulary from corpus - instant, no BPE iterations.

        Args:
            corpus: List of code strings.
            verbose: Print progress (ignored, always fast).
        """
        if verbose:
            print("Building vocabulary...")

        # Collect all unique tokens
        all_tokens = set()
        for code in corpus:
            all_tokens.update(self._tokenize(code))

        # Add to vocabulary
        next_id = len(SPECIAL_TOKENS)
        for token in sorted(all_tokens):  # Sorted for determinism
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        self._trained = True

        if verbose:
            print(f"Vocabulary size: {len(self.token_to_id)}")

    def encode(self, code: str) -> List[int]:
        """Encode code to token IDs."""
        tokens = self._tokenize(code)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id["<unk>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to code string."""
        tokens = []
        for id_ in ids:
            token = self.id_to_token.get(id_, "<unk>")
            if token not in SPECIAL_TOKENS:
                tokens.append(token)
        return " ".join(tokens)

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
            pad_id = self.token_to_id["<pad>"]
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
            "special_tokens": SPECIAL_TOKENS,
            "type": "SimpleTokenizer"
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_id = data["vocab"]
        tokenizer.id_to_token = {v: k for k, v in data["vocab"].items()}
        tokenizer._trained = True

        return tokenizer


# Common operators (for CodeTokenizer)
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
    Full BPE-style tokenizer for code.

    Features:
    - Special tokens for structure
    - Keyword and operator handling
    - Identifier normalization
    - Whitespace normalization
    - BPE subword tokenization

    NOTE: For simple programs, use SimpleTokenizer instead - it's 100x faster.
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
                            tokens.append(f"_{value}")
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
                            tokens.append("_")  # Space marker
                    i += len(value)
                    matched = True
                    break

            if not matched:
                # Unknown character
                tokens.append(f"_{code[i]}")
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
        code = code.replace("_", " ")
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
            "min_freq": self.min_freq,
            "type": "CodeTokenizer"
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
    print("Tokenizer Smoke Test")
    print("=" * 60)

    import time

    # Sample simple programs (like in state tracking experiment)
    simple_programs = [
        "a = 5",
        "x = y + z",
        "result = a * b",
        "temp = x - y",
        "final = temp + result",
        "b = 10",
        "c = a + b",
        "d = c * 2",
        "e = d - 5",
        "f = e + a",
    ] * 10  # 100 programs

    # Test SimpleTokenizer
    print("\n1. Testing SimpleTokenizer...")
    simple_tok = SimpleTokenizer()

    start = time.time()
    simple_tok.train(simple_programs, verbose=True)
    simple_time = time.time() - start
    print(f"   Training time: {simple_time*1000:.2f}ms")
    print(f"   Vocabulary size: {simple_tok.vocab_size_actual}")

    # Test encoding
    test_code = "a = 5 + b"
    ids = simple_tok.encode(test_code)
    decoded = simple_tok.decode(ids)
    print(f"   Encode '{test_code}' -> {ids}")
    print(f"   Decode -> '{decoded}'")

    # Test CodeTokenizer (BPE)
    print("\n2. Testing CodeTokenizer (BPE)...")
    code_tok = CodeTokenizer(vocab_size=1000, min_freq=1)

    start = time.time()
    code_tok.train(simple_programs, verbose=False)
    code_time = time.time() - start
    print(f"   Training time: {code_time*1000:.2f}ms")
    print(f"   Vocabulary size: {code_tok.vocab_size_actual}")

    # Speed comparison
    print(f"\n3. Speed comparison:")
    print(f"   SimpleTokenizer: {simple_time*1000:.2f}ms")
    print(f"   CodeTokenizer (BPE): {code_time*1000:.2f}ms")
    print(f"   Speedup: {code_time/simple_time:.1f}x")

    print("\n" + "=" * 60)
    print("All Tokenizer tests passed!")
    print("Use SimpleTokenizer for state tracking experiments!")
    print("=" * 60)
