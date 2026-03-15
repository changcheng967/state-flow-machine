"""
Graph Attention for System 3 (Structure)

Dynamic graph neural network for code structure modeling.
Nodes = functions, classes, variables, files.
Edges = calls, imports, mutates, reads.

Uses sparse message-passing for efficiency.
Updated incrementally as code changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math


class EdgeTypeEmbedding(nn.Module):
    """
    Learnable embeddings for edge types in code graphs.

    Edge types: calls, imports, mutates, reads, defines, contains
    """

    def __init__(self, num_edge_types: int, edge_dim: int):
        """
        Initialize edge type embeddings.

        Args:
            num_edge_types: Number of different edge types.
            edge_dim: Dimension of edge embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_edge_types, edge_dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Get edge type embeddings.

        Args:
            edge_types: Tensor of edge type indices (batch, num_edges).

        Returns:
            Edge type embeddings (batch, num_edges, edge_dim).
        """
        return self.embedding(edge_types)


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer with multi-head attention.

    Computes attention-weighted aggregation of neighbor messages.
    Supports different edge types with learned relation weights.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize graph attention layer.

        Args:
            node_dim: Dimension of node features.
            edge_dim: Dimension of edge features.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.v_proj = nn.Linear(node_dim, node_dim, bias=False)

        # Edge projection
        self.edge_proj = nn.Linear(edge_dim, node_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(node_dim, node_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for graph attention.

        Args:
            node_features: Node features (batch, num_nodes, node_dim).
            edge_index: Edge indices (2, num_edges) with [source, target].
            edge_features: Edge features (batch, num_edges, edge_dim).
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (updated_node_features, attention_weights).
        """
        batch_size, num_nodes, _ = node_features.shape
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # No edges, return unchanged
            if return_attention:
                return node_features, None
            return node_features, None

        # Get source and target nodes
        source_idx = edge_index[0]  # (num_edges,)
        target_idx = edge_index[1]  # (num_edges,)

        # Project nodes
        q = self.q_proj(node_features)  # (batch, num_nodes, node_dim)
        k = self.k_proj(node_features)  # (batch, num_nodes, node_dim)
        v = self.v_proj(node_features)  # (batch, num_nodes, node_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Get edge features
        edge_proj = self.edge_proj(edge_features)  # (batch, num_edges, node_dim)
        edge_proj = edge_proj.view(batch_size, num_edges, self.num_heads, self.head_dim)
        edge_proj = edge_proj.permute(0, 2, 1, 3)  # (batch, heads, edges, head_dim)

        # Gather source and target features for each edge
        # q_target: queries from target nodes
        # k_source: keys from source nodes
        # v_source: values from source nodes

        # Expand indices for batch and heads
        target_idx_exp = target_idx.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.num_heads, -1
        )  # (batch, heads, num_edges)
        source_idx_exp = source_idx.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.num_heads, -1
        )

        # Gather features
        q_edge = torch.gather(q, 2, target_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        k_edge = torch.gather(k, 2, source_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        v_edge = torch.gather(v, 2, source_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        # Compute attention scores
        # attention = Q_target @ (K_source + E)^T
        attn_input = k_edge + edge_proj  # (batch, heads, edges, head_dim)
        attn_scores = (q_edge * attn_input).sum(dim=-1) / math.sqrt(self.head_dim)
        # (batch, heads, edges)

        # Softmax over edges pointing to same target node (scatter-based softmax)
        # Without this, all edges share one softmax — edges to popular nodes get crushed
        # by edges to unpopular nodes, breaking per-node attention semantics.
        num_inf = float('-inf')
        # Step 1: Find max score per target node (for numerical stability)
        max_per_target = torch.full((batch_size, self.num_heads, num_nodes), num_inf,
                                     device=device, dtype=attn_scores.dtype)
        max_per_target.scatter_reduce_(2, target_idx_exp.unsqueeze(-1), attn_scores, reduce='amax', include_self=False)
        # Step 2: Subtract max (numerical stability)
        attn_stable = attn_scores - max_per_target.gather(2, target_idx_exp.unsqueeze(-1))
        # Step 3: Exp
        exp_scores = torch.exp(attn_stable)
        # Step 4: Sum exponentials per target node
        sum_per_target = torch.zeros(batch_size, self.num_heads, num_nodes, device=device, dtype=exp_scores.dtype)
        sum_per_target.scatter_add_(2, target_idx_exp.unsqueeze(-1), exp_scores)
        # Step 5: Normalize
        attn_weights = exp_scores / sum_per_target.gather(2, target_idx_exp.unsqueeze(-1)).clamp(min=1e-7)
        attn_weights = self.dropout(attn_weights)

        # Aggregate messages
        # For each target node, sum attention-weighted source values
        messages = attn_weights.unsqueeze(-1) * v_edge  # (batch, heads, edges, head_dim)

        # Scatter add to target nodes
        aggregated = torch.zeros_like(q)  # (batch, heads, nodes, head_dim)
        target_idx_scatter = target_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        aggregated.scatter_add_(2, target_idx_scatter, messages)

        # Reshape and project
        aggregated = aggregated.transpose(1, 2).contiguous()
        aggregated = aggregated.view(batch_size, num_nodes, self.node_dim)
        output = self.out_proj(aggregated)

        # Residual and layer norm
        output = self.layer_norm(node_features + self.dropout(output))

        if return_attention:
            return output, attn_weights
        return output, None


class GraphAttentionNetwork(nn.Module):
    """
    Multi-layer Graph Attention Network for code structure.

    Processes code graphs with sparse message-passing.
    Supports incremental updates for dynamic graphs.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize GAT.

        Args:
            node_dim: Node feature dimension.
            edge_dim: Edge feature dimension.
            num_layers: Number of attention layers.
            num_heads: Number of attention heads per layer.
            dropout: Dropout probability.
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        # Graph attention layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(node_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 4, node_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through all layers.

        Args:
            node_features: (batch, num_nodes, node_dim)
            edge_index: (2, num_edges)
            edge_features: (batch, num_edges, edge_dim)
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (updated_node_features, list_of_attention_weights).
        """
        attentions = [] if return_attention else None

        x = node_features
        for i, (gat_layer, ff_layer, ff_norm) in enumerate(
            zip(self.layers, self.ff_layers, self.ff_norms)
        ):
            # Graph attention
            x, attn = gat_layer(x, edge_index, edge_features, return_attention)
            if return_attention:
                attentions.append(attn)

            # Feed-forward with residual
            x = x + ff_layer(ff_norm(x))

        return x, attentions


class DynamicGraphUpdater(nn.Module):
    """
    Incremental graph updater for efficient updates when code changes.

    Instead of recomputing the entire graph, only updates affected nodes
    and their neighbors.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize dynamic updater.

        Args:
            node_dim: Node feature dimension.
            edge_dim: Edge feature dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.gat = GraphAttentionLayer(node_dim, edge_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        changed_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Update only changed nodes and their neighbors.

        Args:
            node_features: All node features (batch, num_nodes, node_dim).
            edge_index: Edge indices (2, num_edges).
            edge_features: Edge features (batch, num_edges, edge_dim).
            changed_nodes: Indices of changed nodes (num_changed,).

        Returns:
            Updated node features (batch, num_nodes, node_dim).
        """
        # Find affected nodes (changed + neighbors)
        affected_mask = torch.zeros(node_features.size(1), dtype=torch.bool, device=node_features.device)
        affected_mask[changed_nodes] = True

        # Find neighbors
        for node_idx in changed_nodes:
            # Outgoing edges
            targets = edge_index[1, edge_index[0] == node_idx]
            affected_mask[targets] = True
            # Incoming edges
            sources = edge_index[0, edge_index[1] == node_idx]
            affected_mask[sources] = True

        affected_indices = torch.where(affected_mask)[0]

        if len(affected_indices) == 0:
            return node_features

        # Create subgraph for affected nodes
        node_mapping = {old.item(): new for new, old in enumerate(affected_indices)}

        # Filter edges where both endpoints are affected
        sub_edge_mask = affected_mask[edge_index[0]] & affected_mask[edge_index[1]]
        sub_edge_index = edge_index[:, sub_edge_mask]

        # Remap edge indices
        sub_edge_index_remapped = torch.zeros_like(sub_edge_index)
        for i in range(sub_edge_index.size(1)):
            sub_edge_index_remapped[0, i] = node_mapping[sub_edge_index[0, i].item()]
            sub_edge_index_remapped[1, i] = node_mapping[sub_edge_index[1, i].item()]

        sub_edge_features = edge_features[:, sub_edge_mask, :]

        # Extract subgraph features
        sub_features = node_features[:, affected_indices, :]

        # Update subgraph
        updated_sub, _ = self.gat(sub_features, sub_edge_index_remapped, sub_edge_features)
        updated_sub = updated_sub + self.ff(self.norm(updated_sub))

        # Scatter back to full graph
        updated_features = node_features.clone()
        updated_features[:, affected_indices, :] = updated_sub

        return updated_features


class CodeGraphNodeEncoder(nn.Module):
    """
    Encodes different code element types as graph nodes.

    Element types: function, class, variable, file, module, statement
    """

    def __init__(
        self,
        input_dim: int,
        node_dim: int,
        num_element_types: int = 6
    ):
        """
        Initialize node encoder.

        Args:
            input_dim: Dimension of input features (e.g., from tokenizer).
            node_dim: Output node dimension.
            num_element_types: Number of code element types.
        """
        super().__init__()

        self.type_embedding = nn.Embedding(num_element_types, node_dim)
        self.feature_proj = nn.Linear(input_dim, node_dim, bias=False)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        features: torch.Tensor,
        element_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode code elements as graph nodes.

        Args:
            features: Input features (batch, num_nodes, input_dim).
            element_types: Element type indices (batch, num_nodes).

        Returns:
            Node embeddings (batch, num_nodes, node_dim).
        """
        type_emb = self.type_embedding(element_types)  # (batch, num_nodes, node_dim)
        feat_proj = self.feature_proj(features)  # (batch, num_nodes, node_dim)
        return self.layer_norm(type_emb + feat_proj)


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Graph Attention Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 20
    num_edges = 40
    node_dim = 64
    edge_dim = 32
    num_heads = 4
    num_edge_types = 6

    # Test EdgeTypeEmbedding
    print("\n1. Testing EdgeTypeEmbedding...")
    edge_emb = EdgeTypeEmbedding(num_edge_types, edge_dim)
    edge_types = torch.randint(0, num_edge_types, (batch_size, num_edges))
    edge_features = edge_emb(edge_types)
    print(f"   Edge types shape: {edge_types.shape}")
    print(f"   Edge features shape: {edge_features.shape}")
    assert edge_features.shape == (batch_size, num_edges, edge_dim), "Edge features shape mismatch!"
    print("   ✓ EdgeTypeEmbedding test passed!")

    # Create random graph
    node_features = torch.randn(batch_size, num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Test GraphAttentionLayer
    print("\n2. Testing GraphAttentionLayer...")
    gat_layer = GraphAttentionLayer(node_dim, edge_dim, num_heads)
    updated, attn = gat_layer(node_features, edge_index, edge_features, return_attention=True)
    print(f"   Input shape: {node_features.shape}")
    print(f"   Output shape: {updated.shape}")
    assert updated.shape == (batch_size, num_nodes, node_dim), "GAT layer output shape mismatch!"
    print("   ✓ GraphAttentionLayer test passed!")

    # Test GraphAttentionNetwork
    print("\n3. Testing GraphAttentionNetwork...")
    gat = GraphAttentionNetwork(node_dim, edge_dim, num_layers=3, num_heads=num_heads)
    output, attentions = gat(node_features, edge_index, edge_features, return_attention=True)
    print(f"   Output shape: {output.shape}")
    print(f"   Number of attention layers: {len(attentions) if attentions else 0}")
    assert output.shape == (batch_size, num_nodes, node_dim), "GAT output shape mismatch!"
    print("   ✓ GraphAttentionNetwork test passed!")

    # Test DynamicGraphUpdater
    print("\n4. Testing DynamicGraphUpdater...")
    updater = DynamicGraphUpdater(node_dim, edge_dim, num_heads)
    changed_nodes = torch.tensor([0, 5, 10])
    updated_features = updater(node_features, edge_index, edge_features, changed_nodes)
    print(f"   Updated features shape: {updated_features.shape}")
    assert updated_features.shape == (batch_size, num_nodes, node_dim), "Updater output shape mismatch!"
    print("   ✓ DynamicGraphUpdater test passed!")

    # Test CodeGraphNodeEncoder
    print("\n5. Testing CodeGraphNodeEncoder...")
    encoder = CodeGraphNodeEncoder(input_dim=32, node_dim=node_dim)
    input_features = torch.randn(batch_size, num_nodes, 32)
    element_types = torch.randint(0, 6, (batch_size, num_nodes))
    node_emb = encoder(input_features, element_types)
    print(f"   Input features shape: {input_features.shape}")
    print(f"   Node embeddings shape: {node_emb.shape}")
    assert node_emb.shape == (batch_size, num_nodes, node_dim), "Node encoder output shape mismatch!"
    print("   ✓ CodeGraphNodeEncoder test passed!")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for param in gat.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in gat.parameters())
    print(f"\n   Total parameters (GAT): {total_params:,}")

    print("\n" + "=" * 60)
    print("All Graph Attention tests passed!")
    print("=" * 60)
