"""
System 3: Structure

Dynamic graph neural network for code structure modeling.
Nodes = functions, classes, variables, files.
Edges = calls, imports, mutates, reads.

Updated via sparse message-passing. Gives the model a live dependency map
so it doesn't break existing code when editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

import sys
sys.path.insert(0, str(__file__).rsplit('sfm', 1)[0])
from sfm.components.graph_attention import (
    GraphAttentionNetwork,
    DynamicGraphUpdater,
    CodeGraphNodeEncoder,
    EdgeTypeEmbedding
)


# Node types for code elements
NODE_TYPES = {
    "function": 0,
    "class": 1,
    "variable": 2,
    "file": 3,
    "module": 4,
    "statement": 5,
}

# Edge types for code relationships
EDGE_TYPES = {
    "calls": 0,      # Function calls function
    "imports": 1,    # Module imports module
    "mutates": 2,    # Statement modifies variable
    "reads": 3,      # Statement reads variable
    "defines": 4,    # Scope defines symbol
    "contains": 5,   # Parent contains child
}


class CodeGraph:
    """
    Represents the code structure as a graph.

    Tracks nodes (code elements) and edges (relationships).
    Supports incremental updates.
    """

    def __init__(
        self,
        max_nodes: int = 1024,
        max_edges: int = 4096,
        num_edge_types: int = 6
    ):
        """
        Initialize empty code graph.

        Args:
            max_nodes: Maximum number of nodes.
            max_edges: Maximum number of edges.
            num_edge_types: Number of edge types.
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.num_edge_types = num_edge_types

        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int, int]] = []  # (source, target, edge_type)
        self.node_to_idx: Dict[str, int] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        features: Optional[torch.Tensor] = None
    ) -> int:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of node (function, class, etc.).
            features: Optional node features.

        Returns:
            Node index.
        """
        if node_id in self.node_to_idx:
            return self.node_to_idx[node_id]

        idx = len(self.nodes)
        if idx >= self.max_nodes:
            raise ValueError(f"Maximum nodes ({self.max_nodes}) exceeded")

        self.node_to_idx[node_id] = idx
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "type_idx": NODE_TYPES.get(node_type, 5),
            "features": features
        })

        return idx

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str
    ) -> bool:
        """
        Add an edge to the graph.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of relationship.

        Returns:
            True if edge was added.
        """
        if source_id not in self.node_to_idx or target_id not in self.node_to_idx:
            return False

        if len(self.edges) >= self.max_edges:
            raise ValueError(f"Maximum edges ({self.max_edges}) exceeded")

        source_idx = self.node_to_idx[source_id]
        target_idx = self.node_to_idx[target_id]
        edge_type_idx = EDGE_TYPES.get(edge_type, 0)

        # Check for duplicate
        edge = (source_idx, target_idx, edge_type_idx)
        if edge not in self.edges:
            self.edges.append(edge)

        return True

    def to_tensor(
        self,
        device: torch.device,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert graph to tensors for GNN processing.

        Args:
            device: Target device.
            batch_size: Batch size for batching.

        Returns:
            Tuple of (node_types, edge_index, edge_types).
        """
        num_nodes = len(self.nodes)

        if num_nodes == 0:
            # Empty graph
            node_types = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_types = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return node_types, edge_index, edge_types

        # Node types
        node_types_list = [node["type_idx"] for node in self.nodes]
        node_types = torch.tensor(node_types_list, dtype=torch.long, device=device)
        node_types = node_types.unsqueeze(0).expand(batch_size, -1)

        # Edge index
        if len(self.edges) > 0:
            edge_index = torch.tensor(
                [(e[0], e[1]) for e in self.edges],
                dtype=torch.long, device=device
            ).T  # (2, num_edges)
            edge_types_list = [e[2] for e in self.edges]
            edge_types = torch.tensor(edge_types_list, dtype=torch.long, device=device)
            edge_types = edge_types.unsqueeze(0).expand(batch_size, -1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_types = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        return node_types, edge_index, edge_types

    def get_neighbors(self, node_idx: int, edge_type: Optional[str] = None) -> List[int]:
        """
        Get neighbors of a node.

        Args:
            node_idx: Node index.
            edge_type: Optional filter by edge type.

        Returns:
            List of neighbor node indices.
        """
        neighbors = []
        edge_type_idx = EDGE_TYPES.get(edge_type) if edge_type else None

        for source, target, etype in self.edges:
            if source == node_idx:
                if edge_type_idx is None or etype == edge_type_idx:
                    neighbors.append(target)
            if target == node_idx:
                if edge_type_idx is None or etype == edge_type_idx:
                    neighbors.append(source)

        return list(set(neighbors))


class StructureSystem(nn.Module):
    """
    System 3: Structure

    Dynamic graph neural network for code structure.

    Features:
    - Nodes for functions, classes, variables, files
    - Edges for calls, imports, mutates, reads, defines, contains
    - Sparse message-passing for efficiency
    - Incremental updates for dynamic graphs
    """

    def __init__(
        self,
        input_dim: int,
        node_dim: int = 256,
        edge_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_nodes: int = 1024,
        max_edges: int = 4096,
        dropout: float = 0.1
    ):
        """
        Initialize structure system.

        Args:
            input_dim: Input dimension from perception.
            node_dim: Node feature dimension.
            edge_dim: Edge feature dimension.
            num_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            max_nodes: Maximum number of nodes.
            max_edges: Maximum number of edges.
            dropout: Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        # Node encoder
        self.node_encoder = CodeGraphNodeEncoder(
            input_dim=input_dim,
            node_dim=node_dim,
            num_element_types=len(NODE_TYPES)
        )

        # Edge type embeddings
        self.edge_embedding = EdgeTypeEmbedding(
            num_edge_types=len(EDGE_TYPES),
            edge_dim=edge_dim
        )

        # Graph attention network
        self.gat = GraphAttentionNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Dynamic updater for incremental changes
        self.dynamic_updater = DynamicGraphUpdater(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Output projection back to input_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, input_dim),
            nn.Dropout(dropout)
        )

        # Global graph representation
        self.global_pool = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.Tanh()
        )

    def build_graph_from_sequence(
        self,
        x: torch.Tensor,
        token_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a graph from sequence input (simplified).

        Creates nodes for each token and edges based on locality.

        Args:
            x: Input sequence (batch, seq_len, input_dim).
            token_types: Optional token type indices.

        Returns:
            Tuple of (node_features, edge_index, edge_features).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Each token becomes a node
        node_features = x  # (batch, seq_len, input_dim)

        # Default token types
        if token_types is None:
            token_types = torch.full((batch_size, seq_len), 5, dtype=torch.long, device=device)

        # Build edges based on locality (simplified)
        # Connect each token to its neighbors within a window
        edges = []
        window = 3

        for i in range(seq_len):
            for j in range(max(0, i - window), min(seq_len, i + window + 1)):
                if i != j:
                    edge_type = 5  # "contains" as default
                    edges.append((i, j, edge_type))

        if len(edges) > 0:
            edge_index = torch.tensor(
                [(e[0], e[1]) for e in edges],
                dtype=torch.long, device=device
            ).T  # (2, num_edges)
            edge_types = torch.tensor(
                [e[2] for e in edges],
                dtype=torch.long, device=device
            )
            edge_types = edge_types.unsqueeze(0).expand(batch_size, -1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_types = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        return node_features, edge_index, edge_types

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[CodeGraph] = None,
        return_graph_output: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through structure system.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            graph: Optional pre-built CodeGraph.
            return_graph_output: Whether to return graph-level output.

        Returns:
            Output tensor (batch, seq_len, input_dim).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Build or use provided graph
        if graph is not None and len(graph.nodes) > 0:
            node_types, edge_index, edge_types = graph.to_tensor(device, batch_size)

            # Create node features (zeros if not provided)
            if len(graph.nodes) > 0 and graph.nodes[0].get("features") is not None:
                node_features = torch.stack([
                    n["features"] for n in graph.nodes if n.get("features") is not None
                ], dim=0)
                if node_features.dim() == 2:
                    node_features = node_features.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Use sequence features for nodes
                num_nodes = min(len(graph.nodes), seq_len)
                node_features = x[:, :num_nodes, :]
                node_types = node_types[:, :num_nodes]
        else:
            # Build graph from sequence
            node_features, edge_index, edge_types = self.build_graph_from_sequence(x)
            node_types = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        num_nodes = node_features.size(1)

        if num_nodes == 0:
            return x  # No nodes, return unchanged

        # Encode nodes
        node_embeddings = self.node_encoder(node_features, node_types)

        # Encode edges
        edge_features = self.edge_embedding(edge_types)

        # Process through GNN
        updated_nodes, _ = self.gat(
            node_embeddings,
            edge_index,
            edge_features
        )

        # Project back to input dimension
        output = self.output_proj(updated_nodes)

        # Handle sequence length mismatch
        if output.size(1) < seq_len:
            # Pad output
            padding = torch.zeros(batch_size, seq_len - output.size(1), self.input_dim, device=device)
            output = torch.cat([output, padding], dim=1)
        elif output.size(1) > seq_len:
            # Truncate to match input
            output = output[:, :seq_len, :]

        # Add residual connection
        output = output + x

        if return_graph_output:
            # Compute global graph representation
            graph_repr = self.global_pool(updated_nodes.mean(dim=1))
            return output, graph_repr

        return output

    def update_graph(
        self,
        graph: CodeGraph,
        changed_nodes: List[int],
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Incrementally update graph for changed nodes.

        Args:
            graph: The code graph.
            changed_nodes: Indices of changed nodes.
            node_features: Current node features.

        Returns:
            Updated node features.
        """
        if len(changed_nodes) == 0:
            return node_features

        _, edge_index, edge_types = graph.to_tensor(node_features.device, node_features.size(0))
        edge_features = self.edge_embedding(edge_types)

        changed_tensor = torch.tensor(changed_nodes, dtype=torch.long, device=node_features.device)

        updated = self.dynamic_updater(
            node_features,
            edge_index,
            edge_features,
            changed_tensor
        )

        return updated

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Structure System Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 32
    input_dim = 128
    node_dim = 64
    edge_dim = 32

    # Initialize structure system
    print("\n1. Initializing StructureSystem...")
    structure = StructureSystem(
        input_dim=input_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_layers=2,
        num_heads=4,
        max_nodes=256,
        max_edges=512
    )

    # Test CodeGraph
    print("\n2. Testing CodeGraph...")
    graph = CodeGraph(max_nodes=100, max_edges=200)

    # Add nodes
    graph.add_node("func_main", "function")
    graph.add_node("var_x", "variable")
    graph.add_node("class_Foo", "class")

    # Add edges
    graph.add_edge("func_main", "var_x", "defines")
    graph.add_edge("class_Foo", "func_main", "contains")

    print(f"   Number of nodes: {len(graph.nodes)}")
    print(f"   Number of edges: {len(graph.edges)}")

    # Get neighbors
    neighbors = graph.get_neighbors(0)  # func_main
    print(f"   Neighbors of func_main: {neighbors}")

    # Convert to tensor
    node_types, edge_index, edge_types = graph.to_tensor(torch.device('cpu'), batch_size=2)
    print(f"   Node types shape: {node_types.shape}")
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge types shape: {edge_types.shape}")
    print("   ✓ CodeGraph test passed!")

    # Test forward pass without graph
    print("\n3. Testing forward pass (auto graph)...")
    x = torch.randn(batch_size, seq_len, input_dim)
    output = structure(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("   ✓ Forward pass test passed!")

    # Test forward pass with graph
    print("\n4. Testing forward pass with CodeGraph...")
    output, graph_repr = structure(x, graph=graph, return_graph_output=True)
    print(f"   Output shape: {output.shape}")
    print(f"   Graph representation shape: {graph_repr.shape}")
    assert output.shape == x.shape, "Output with graph shape mismatch!"
    assert graph_repr.shape == (batch_size, node_dim), "Graph repr shape mismatch!"
    print("   ✓ Forward pass with graph test passed!")

    # Test incremental update
    print("\n5. Testing incremental update...")
    node_features = torch.randn(batch_size, len(graph.nodes), node_dim)
    changed_nodes = [0, 1]  # func_main and var_x changed
    updated = structure.update_graph(graph, changed_nodes, node_features)
    print(f"   Updated features shape: {updated.shape}")
    assert updated.shape == node_features.shape, "Updated features shape mismatch!"
    print("   ✓ Incremental update test passed!")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    loss = output.sum()
    loss.backward()
    grad_exists = False
    for param in structure.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break
    assert grad_exists, "No gradients found!"
    print("   ✓ Gradient flow test passed!")

    # Count parameters
    total_params = structure.count_parameters()
    print(f"\n   Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All Structure System tests passed!")
    print("=" * 60)
