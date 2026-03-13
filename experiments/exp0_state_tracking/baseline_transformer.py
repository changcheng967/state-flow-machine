"""
Baseline Transformer for State Tracking Experiment

Standard transformer encoder-decoder for comparison with State Slots.
This baseline should fail to generalize to longer sequences (TC0 limit).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Standard transformer baseline for comparison.

    Uses standard O(n²) attention, which cannot efficiently track
    state through long sequences due to TC0 circuit complexity limits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_output_classes: int = 1000
    ):
        """
        Initialize transformer baseline.

        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            d_ff: Feed-forward dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            num_output_classes: Number of output classes (for classification).
        """
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output heads
        self.classification_head = nn.Linear(d_model, num_output_classes)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source tokens (batch, seq_len).
            src_mask: Optional attention mask.

        Returns:
            Encoded representation (batch, seq_len, d_model).
        """
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: Target tokens (batch, tgt_len).
            memory: Encoder output.
            tgt_mask: Causal mask for target.
            memory_mask: Cross attention mask.

        Returns:
            Decoded output (batch, tgt_len, d_model).
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Causal mask
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask
        )
        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source tokens (batch, seq_len).
            tgt: Target tokens for decoding (optional).
            src_mask: Attention mask for source.
            task: "classification" or "generation".

        Returns:
            Output logits.
        """
        # Encode
        memory = self.encode(src, src_mask)

        if task == "classification":
            # Use [CLS] token (first position) for classification
            cls_output = memory[:, 0, :]
            logits = self.classification_head(cls_output)
        elif task == "generation" and tgt is not None:
            output = self.decode(tgt, memory, memory_mask=src_mask)
            logits = self.lm_head(output)
        else:
            # Default: use pooled representation
            pooled = memory.mean(dim=1)
            logits = self.classification_head(pooled)

        return logits

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class TransformerEncoderOnly(nn.Module):
    """
    Encoder-only transformer for simpler baseline comparison.

    Uses pooled output for classification/regression.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_output_classes: int = 1000
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_output_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tokens (batch, seq_len).
            mask: Attention mask.
            pooling: Pooling method ("mean", "first", "last").

        Returns:
            Output logits (batch, num_classes).
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        x = self.encoder(x, src_key_padding_mask=mask)

        if pooling == "mean":
            if mask is not None:
                mask_exp = (~mask).unsqueeze(-1).float()
                x = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
            else:
                x = x.mean(dim=1)
        elif pooling == "first":
            x = x[:, 0, :]
        elif pooling == "last":
            x = x[:, -1, :]

        return self.output_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Baseline Transformer Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    d_model = 128

    # Test TransformerBaseline
    print("\n1. Testing TransformerBaseline...")
    model = TransformerBaseline(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        num_output_classes=100
    )

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, 10))

    # Classification task
    logits = model(src, task="classification")
    print(f"   Classification logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 100), "Classification shape mismatch!"

    # Generation task
    logits = model(src, tgt=tgt, task="generation")
    print(f"   Generation logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 10, vocab_size), "Generation shape mismatch!"

    print(f"   Parameters: {model.count_parameters():,}")
    print("   ✓ TransformerBaseline test passed!")

    # Test TransformerEncoderOnly
    print("\n2. Testing TransformerEncoderOnly...")
    encoder = TransformerEncoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_output_classes=100
    )

    logits = encoder(src, pooling="mean")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 100), "Encoder shape mismatch!"

    print(f"   Parameters: {encoder.count_parameters():,}")
    print("   ✓ TransformerEncoderOnly test passed!")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = logits.sum()
    loss.backward()
    print("   ✓ Gradient flow test passed!")

    # Compare parameter counts
    print("\n4. Parameter comparison...")
    print(f"   TransformerBaseline (enc+dec): {model.count_parameters():,}")
    print(f"   TransformerEncoderOnly: {encoder.count_parameters():,}")

    print("\n" + "=" * 60)
    print("All Baseline Transformer tests passed!")
    print("=" * 60)
