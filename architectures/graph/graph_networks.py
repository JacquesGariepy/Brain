"""
Graph Neural Networks - SOTA

Implementations:
- GCN (Graph Convolutional Networks)
- GAT (Graph Attention Networks)
- GraphSAGE
- GIN (Graph Isomorphism Network)
- Temporal Graph Networks
- Message Passing Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.

    Performs neighborhood aggregation via convolution on graphs.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            edge_weight: Optional edge weights

        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Normalize adjacency matrix
        if edge_weight is None:
            # Add self-loops
            adj = adj + torch.eye(adj.size(0), device=adj.device)

            # Compute normalization: D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        else:
            norm = adj * edge_weight.unsqueeze(0)

        # Aggregate: AXW
        support = self.linear(x)
        output = torch.matmul(norm, support)

        return output


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.

    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transformations per head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features, 1))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with multi-head attention.

        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Updated features (num_nodes, out_features * num_heads) if concat
            else (num_nodes, out_features)
        """
        num_nodes = x.size(0)

        # Linear transformation for each head
        h = torch.stack([torch.matmul(x, self.W[i]) for i in range(self.num_heads)])
        # h: (num_heads, num_nodes, out_features)

        # Compute attention scores
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (num_heads, num_nodes, num_nodes, out_features)
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # (num_heads, num_nodes, num_nodes, out_features)

        # Concatenate and compute attention
        e = torch.cat([h_i, h_j], dim=-1)  # (num_heads, num_nodes, num_nodes, 2*out_features)
        e = self.leakyrelu(torch.matmul(e, self.a).squeeze(-1))  # (num_heads, num_nodes, num_nodes)

        # Mask attention for non-neighbors
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)

        # Softmax
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # Aggregate
        h_prime = torch.matmul(attention, h)  # (num_heads, num_nodes, out_features)

        # Concatenate or average heads
        if self.concat:
            output = h_prime.transpose(0, 1).contiguous().view(num_nodes, -1)
        else:
            output = h_prime.mean(dim=0)

        return output


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with neighborhood sampling.

    Efficient for large graphs via sampling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = 'mean',
        normalize: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.normalize = normalize

        # Two linear layers: one for self, one for neighbors
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            adj: Adjacency matrix

        Returns:
            Updated node features
        """
        # Aggregate neighbors
        if self.aggregator == 'mean':
            # Mean aggregation
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1  # Avoid division by zero
            neigh_features = torch.matmul(adj, x) / deg
        elif self.aggregator == 'max':
            # Max aggregation
            neigh_features = []
            for i in range(x.size(0)):
                neighbors = adj[i].nonzero(as_tuple=True)[0]
                if len(neighbors) > 0:
                    neigh_features.append(x[neighbors].max(dim=0)[0])
                else:
                    neigh_features.append(torch.zeros(self.in_features, device=x.device))
            neigh_features = torch.stack(neigh_features)
        elif self.aggregator == 'lstm':
            # LSTM aggregation (not implemented in this simplified version)
            raise NotImplementedError("LSTM aggregator not implemented")
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Combine self and neighbor features
        self_features = self.linear_self(x)
        neigh_features = self.linear_neigh(neigh_features)

        output = self_features + neigh_features

        # Normalize
        if self.normalize:
            output = F.normalize(output, p=2, dim=-1)

        return F.relu(output)


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.

    Provably maximally expressive GNN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        epsilon: float = 0.0,
        learn_epsilon: bool = False
    ):
        super().__init__()

        if learn_epsilon:
            self.epsilon = nn.Parameter(torch.tensor(epsilon))
        else:
            self.register_buffer('epsilon', torch.tensor(epsilon))

        # MLP for update
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with injective aggregation.

        Args:
            x: Node features
            adj: Adjacency matrix

        Returns:
            Updated node features
        """
        # Sum aggregation
        neigh_sum = torch.matmul(adj, x)

        # Combine with self features
        output = (1 + self.epsilon) * x + neigh_sum

        # Apply MLP
        output = self.mlp(output)

        return output


class MessagePassingLayer(nn.Module):
    """
    Generic Message Passing Neural Network layer.

    Flexible framework for various GNN architectures.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: Optional[int] = None
    ):
        super().__init__()

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_features + (edge_dim if edge_dim else 0), out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute messages from node j to node i.

        Args:
            x_i: Target node features
            x_j: Source node features
            edge_attr: Optional edge attributes

        Returns:
            Messages
        """
        if edge_attr is not None:
            inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            inp = torch.cat([x_i, x_j], dim=-1)

        return self.message_mlp(inp)

    def aggregate(self, messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Aggregate messages for each node.

        Args:
            messages: All messages
            index: Target node indices
            num_nodes: Total number of nodes

        Returns:
            Aggregated messages per node
        """
        # Sum aggregation
        aggregated = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
        aggregated = aggregated.index_add(0, index, messages)

        return aggregated

    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """
        Update node features.

        Args:
            x: Current node features
            aggregated: Aggregated messages

        Returns:
            Updated node features
        """
        inp = torch.cat([x, aggregated], dim=-1)
        return self.update_mlp(inp)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Optional edge attributes (num_edges, edge_dim)

        Returns:
            Updated node features
        """
        row, col = edge_index

        # Compute messages
        messages = self.message(x[row], x[col], edge_attr)

        # Aggregate
        aggregated = self.aggregate(messages, row, x.size(0))

        # Update
        output = self.update(x, aggregated)

        return output


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network for dynamic graphs.

    Handles graphs that change over time.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()

        # Memory for each node
        self.memory = nn.Parameter(torch.zeros(1, hidden_dim))

        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Memory updater (GRU)
        self.memory_updater = nn.GRUCell(hidden_dim, hidden_dim)

        # Embedding
        self.node_embedding = nn.Linear(node_features + hidden_dim, hidden_dim)

        # Graph layers
        self.graph_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, edge_features)
            for _ in range(num_layers)
        ])

    def get_updated_memory(
        self,
        node_idx: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Update node memory with new messages.

        Args:
            node_idx: Node indices
            messages: Messages for nodes

        Returns:
            Updated memory
        """
        current_memory = self.memory.expand(len(node_idx), -1)
        new_memory = self.memory_updater(messages, current_memory)
        return new_memory

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass on temporal graph.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features (including timestamps)
            node_idx: Indices of nodes to update

        Returns:
            Updated node representations
        """
        # Compute messages
        row, col = edge_index
        messages = self.message_fn(
            torch.cat([x[row], x[col], edge_attr], dim=-1)
        )

        # Update memory for affected nodes
        unique_nodes = torch.unique(torch.cat([row, col]))
        # Aggregate messages per node
        node_messages = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        node_messages = node_messages.index_add(0, row, messages)

        # Update memory
        updated_memory = self.get_updated_memory(unique_nodes, node_messages[unique_nodes])

        # Combine features with memory
        h = self.node_embedding(torch.cat([x[unique_nodes], updated_memory], dim=-1))

        # Apply graph layers
        for layer in self.graph_layers:
            h = layer(h, edge_index, edge_attr)

        return h


class GraphTransformer(nn.Module):
    """
    Graph Transformer combining graphs with transformer architecture.

    Uses graph structure to guide attention.
    """

    def __init__(
        self,
        node_features: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_features, d_model)

        # Transformer layers with graph bias
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            adj: Adjacency matrix (used as attention bias)

        Returns:
            Node representations
        """
        h = self.node_embedding(x)

        for layer in self.layers:
            h = layer(h, adj)

        return self.norm(h)


class GraphTransformerLayer(nn.Module):
    """Single layer of Graph Transformer"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward with graph-biased attention.

        Args:
            x: Node features (num_nodes, d_model)
            adj: Adjacency matrix for attention bias

        Returns:
            Updated features
        """
        # Create attention mask from adjacency
        # Only attend to connected nodes
        attn_mask = (adj == 0).float() * -1e9

        # Self-attention with graph bias
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(
            x_norm.unsqueeze(0),
            x_norm.unsqueeze(0),
            x_norm.unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0)
        )
        x = x + attn_out.squeeze(0)

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x
