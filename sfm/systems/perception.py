"""
System 1: Perception

Linear-attention decoder. O(n) not O(n²). Reads tokens. That's it.

This system handles the basic token processing with linear complexity,
allowing efficient processing of long code sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.components.linear_attention import CausalLinearAttention, LinearAttentionBlock


class TokenEmbedding(nn.Module):
    """
    Token embedding with learned positional encodings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1
    ):
        """
        Initialize token embedding.

        Args:
            vocab_size: Size of vocabulary.
            d_model: Model dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed tokens with positional encoding.

        Args:
            tokens: Token indices (batch, seq_len).
            positions: Optional position indices (batch, seq_len).

        Returns:
            Embedded tokens (batch, seq_len, d_model).
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        token_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(positions)

        embeddings = (token_emb + pos_emb) * math.sqrt(self.d_model)
        return self.dropout(embeddings)


class PerceptionLayer(nn.Module):
    """
    Single perception layer with causal linear attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1
    ):
        """
        Initialize perception layer.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        ff_dim = ff_dim or (4 * d_model)

        # Causal linear attention
        self.attention = CausalLinearAttention(d_model, num_heads, dropout=dropout)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through layer.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            state: Optional attention state for incremental decoding.
            return_state: Whether to return attention state.

        Returns:
            Output tensor, optionally with state.
        """
        # Attention with residual
        if return_state or state is not None:
            attn_out, new_state = self.attention(self.norm1(x), state=state, return_state=True)
        else:
            attn_out = self.attention(self.norm1(x))
            new_state = None

        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        if return_state:
            return x, new_state
        return x


class PerceptionSystem(nn.Module):
    """
    System 1: Perception

    Linear-attention decoder stack for token processing.
    O(n) complexity enables processing long code sequences.

    Features:
    - Causal masking for autoregressive generation
    - Linear attention for efficiency
    - Optional state for incremental decoding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = None,
        max_seq_len: int = 4096,
        dropout: float = 0.1
    ):
        """
        Initialize perception system.

        Args:
            vocab_size: Size of vocabulary.
            d_model: Model dimension.
            num_layers: Number of attention layers.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embedding
        self.embedding = TokenEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Stack of perception layers
        self.layers = nn.ModuleList([
            PerceptionLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (for language modeling)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights with embedding
        self.output_proj.weight = self.embedding.token_embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        state: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_state: bool = False,
        return_logits: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through perception system.

        Args:
            tokens: Token indices (batch, seq_len).
            state: Optional state dict for incremental decoding.
            return_state: Whether to return attention states.
            return_logits: Whether to return vocabulary logits.

        Returns:
            Output tensor (logits or hidden states), optionally with state.
        """
        # Embed tokens
        x = self.embedding(tokens)

        # Initialize state storage if needed
        new_state = {} if return_state else None

        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_state = state.get(str(i)) if state else None

            if return_state:
                x, layer_new_state = layer(x, state=layer_state, return_state=True)
                new_state[str(i)] = layer_new_state
            else:
                x = layer(x, state=layer_state)

        # Final layer norm
        x = self.final_norm(x)

        # Project to vocabulary
        if return_logits:
            output = self.output_proj(x)
        else:
            output = x

        if return_state:
            return output, new_state
        return output

    def encode(
        self,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode tokens to hidden states (no logits).

        Args:
            tokens: Token indices (batch, seq_len).

        Returns:
            Hidden states (batch, seq_len, d_model).
        """
        return self.forward(tokens, return_logits=False)

    def decode_step(
        self,
        token: torch.Tensor,
        state: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Single-step decoding for generation.

        Args:
            token: Single token (batch, 1).
            state: Attention states from previous steps.

        Returns:
            Tuple of (logits, new_state).
        """
        return self.forward(token, state=state, return_state=True, return_logits=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Prompt tokens (batch, prompt_len).
            max_new_tokens: Maximum number of new tokens.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling probability.

        Returns:
            Generated tokens including prompt (batch, total_len).
        """
        self.eval()
        device = prompt.device

        # Initial forward pass to get state
        logits, state = self.forward(prompt, return_state=True)

        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Get last token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Forward pass for next token
            logits, state = self.decode_step(next_token, state)

        return generated

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Perception System Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    d_model = 128
    num_layers = 4
    num_heads = 4

    # Initialize perception system
    print("\n1. Initializing PerceptionSystem...")
    perception = PerceptionSystem(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=256,
        max_seq_len=512
    )

    # Test embedding
    print("\n2. Testing TokenEmbedding...")
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embeddings = perception.embedding(tokens)
    print(f"   Tokens shape: {tokens.shape}")
    print(f"   Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, seq_len, d_model), "Embedding shape mismatch!"
    print("   ✓ TokenEmbedding test passed!")

    # Test forward pass
    print("\n3. Testing forward pass...")
    logits = perception(tokens)
    print(f"   Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch!"
    print("   ✓ Forward pass test passed!")

    # Test stateful decoding
    print("\n4. Testing stateful decoding...")
    logits, state = perception.forward(tokens, return_state=True)
    print(f"   Number of state layers: {len(state)}")

    # Single step decode
    next_token = torch.randint(0, vocab_size, (batch_size, 1))
    next_logits, new_state = perception.decode_step(next_token, state)
    print(f"   Next token logits shape: {next_logits.shape}")
    assert next_logits.shape == (batch_size, 1, vocab_size), "Decode step shape mismatch!"
    print("   ✓ Stateful decoding test passed!")

    # Test encode method
    print("\n5. Testing encode method...")
    hidden = perception.encode(tokens)
    print(f"   Hidden states shape: {hidden.shape}")
    assert hidden.shape == (batch_size, seq_len, d_model), "Hidden shape mismatch!"
    print("   ✓ Encode test passed!")

    # Test generation (small)
    print("\n6. Testing generation...")
    prompt = torch.randint(0, vocab_size, (batch_size, 5))
    generated = perception.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10)
    print(f"   Prompt shape: {prompt.shape}")
    print(f"   Generated shape: {generated.shape}")
    assert generated.shape == (batch_size, 10), "Generation shape mismatch!"
    print("   ✓ Generation test passed!")

    # Test gradient flow
    print("\n7. Testing gradient flow...")
    loss = logits.sum()
    loss.backward()
    grad_exists = False
    for param in perception.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = perception.count_parameters()
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All Perception System tests passed!")
    print("=" * 60)
