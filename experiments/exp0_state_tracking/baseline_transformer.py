"""
Baseline Transformer for State Tracking Experiment

Standard transformer encoder-only for comparison with State Slots.
This baseline should fail to generalize to longer sequences (TC0 limit).

CUSTOM IMPLEMENTATION - No nn.TransformerEncoder/nn.TransformerEncoderLayer
to ensure compatibility with Ascend NPU (which doesn't support nested tensors).

Also contains StateTrackingWrapper (execution system wrapper for regression)
so both train.py and evaluate.py can import it without pulling in torch_npu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Manual multi-head self-attention using only basic ops.

    No nn.MultiheadAttention - pure matmul/softmax for NPU compatibility.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            mask: Padding mask where True = padding (batch, seq_len).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply padding mask as additive mask (-1e9 for padded positions)
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-1e9'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Manual feedforward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer - NPU compatible.

    No nn.TransformerEncoderLayer - manual implementation.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class TransformerEncoder(nn.Module):
    """
    Custom transformer encoder - stack of layers.

    No nn.TransformerEncoder - manual implementation.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerEncoderOnly(nn.Module):
    """
    Encoder-only transformer for simpler baseline comparison.

    CUSTOM IMPLEMENTATION - No PyTorch transformer internals.
    REGRESSION: Output is single scalar in [0, 1] range.
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
        num_output_classes: int = 1  # REGRESSION: single scalar
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Custom encoder (no nn.TransformerEncoder)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)

        # REGRESSION: Output single scalar with sigmoid
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Forward pass (REGRESSION).

        Args:
            x: Input tokens (batch, seq_len).
            mask: Attention mask where True = padding position (batch, seq_len).
            pooling: Pooling method ("mean", "first", "last").

        Returns:
            Output scalar (batch,) in [0, 1] range.
        """
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Encode with custom transformer
        x = self.encoder(x, mask)

        # Pool
        if pooling == "mean":
            if mask is not None:
                # mask is True for padding, so invert for weighting
                mask_weights = (~mask).unsqueeze(-1).float()
                x = (x * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif pooling == "first":
            x = x[:, 0, :]
        elif pooling == "last":
            if mask is not None:
                # Get last non-padded position
                lengths = (~mask).sum(dim=1) - 1
                batch_indices = torch.arange(x.size(0), device=x.device)
                x = x[batch_indices, lengths]
            else:
                x = x[:, -1, :]

        # REGRESSION: Return (batch,) not (batch, 1)
        return self.regressor(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerBaseline(nn.Module):
    """
    Standard transformer baseline for comparison (encoder-decoder).

    Uses standard O(n²) attention, which cannot efficiently track
    state through long sequences due to TC0 circuit complexity limits.

    CUSTOM IMPLEMENTATION - No PyTorch transformer internals.
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
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Custom encoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)

        # Custom decoder (simplified - just more encoder layers for this experiment)
        self.decoder = TransformerEncoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)

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
        """Encode source sequence."""
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        memory = self.encoder(src_emb, src_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence (simplified - uses encoder-style processing)."""
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Concatenate memory and target for cross-attention simulation
        combined = torch.cat([memory, tgt_emb], dim=1)
        combined = self.decoder(combined, memory_mask)

        # Return only target portion
        tgt_len = tgt.size(1)
        return combined[:, -tgt_len:, :]

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode
        memory = self.encode(src, src_mask)

        if task == "classification":
            # Use mean pooling for classification
            if src_mask is not None:
                mask_weights = (~src_mask).unsqueeze(-1).float()
                pooled = (memory * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp(min=1)
            else:
                pooled = memory.mean(dim=1)
            logits = self.classification_head(pooled)
        elif task == "generation" and tgt is not None:
            output = self.decode(tgt, memory, memory_mask=src_mask)
            logits = self.lm_head(output)
        else:
            # Default: use pooled representation
            if src_mask is not None:
                mask_weights = (~src_mask).unsqueeze(-1).float()
                pooled = (memory * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp(min=1)
            else:
                pooled = memory.mean(dim=1)
            logits = self.classification_head(pooled)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class StateTrackingWrapper(nn.Module):
    """Wraps ExecutionSystem for state tracking REGRESSION.

    Defined here (not in train.py) so evaluate.py can import it
    without pulling in torch_npu via train.py's top-level imports.
    """

    def __init__(
        self,
        execution_system: nn.Module,
        vocab_size: int,
        num_classes: int = 1  # REGRESSION: single scalar output
    ):
        super().__init__()
        self.execution = execution_system

        # Pad vocab_size to multiple of 16 for optimal embedding table access
        padded_vocab_size = ((vocab_size + 15) // 16) * 16
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size

        self.embedding = nn.Embedding(padded_vocab_size, execution_system.input_dim)
        # REGRESSION: Output single scalar, apply sigmoid to constrain to [0, 1]
        self.regressor = nn.Sequential(
            nn.LayerNorm(execution_system.input_dim),
            nn.Linear(execution_system.input_dim, execution_system.input_dim),
            nn.ReLU(),
            nn.Linear(execution_system.input_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.execution(x)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)  # (batch,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Baseline Transformer (Custom NPU-Compatible) Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    d_model = 128

    # Test TransformerEncoderOnly
    print("\n1. Testing TransformerEncoderOnly...")
    encoder = TransformerEncoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_output_classes=100
    )

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, -5:] = True  # Mask last 5 positions

    logits = encoder(src, mask=mask, pooling="mean")
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 100), f"Encoder shape mismatch! Got {logits.shape}"

    print(f"   Parameters: {encoder.count_parameters():,}")
    print("   [OK] TransformerEncoderOnly test passed!")

    # Test without mask
    print("\n2. Testing without mask...")
    logits_no_mask = encoder(src, mask=None, pooling="mean")
    print(f"   Output shape: {logits_no_mask.shape}")
    assert logits_no_mask.shape == (batch_size, 100)
    print("   [OK] No-mask test passed!")

    # Test different pooling
    print("\n3. Testing different pooling methods...")
    for pooling in ["mean", "first", "last"]:
        logits_p = encoder(src, mask=mask, pooling=pooling)
        print(f"   {pooling}: {logits_p.shape}")
        assert logits_p.shape == (batch_size, 100)
    print("   [OK] Pooling test passed!")

    # Test TransformerBaseline
    print("\n4. Testing TransformerBaseline...")
    model = TransformerBaseline(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        num_output_classes=100
    )

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
    print("   [OK] TransformerBaseline test passed!")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    logits = encoder(src, mask=mask)
    loss = logits.sum()
    loss.backward()
    grad_count = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"   Parameters with gradients: {grad_count}/{len(list(encoder.parameters()))}")
    assert grad_count > 0, "No gradients found!"
    print("   [OK] Gradient flow test passed!")

    # Compare parameter counts
    print("\n6. Parameter comparison...")
    print(f"   TransformerBaseline (enc+dec): {model.count_parameters():,}")
    print(f"   TransformerEncoderOnly: {encoder.count_parameters():,}")

    print("\n" + "=" * 60)
    print("All Baseline Transformer (NPU-Compatible) tests passed!")
    print("=" * 60)
