"""
GNN-based Q-Function Learning (Section 2 from the paper).

Implements the message-passing GNN for inter-agent communication:
  h_t^{i,(l)} = UPDATE^(l)(h_t^{i,(l-1)}, AGGREGATE^(l)({MESSAGE^(l)(h_t^{j,(l-1)}) | j in N(i)}))

where:
  - N(i) is the set of neighbors of agent i in communication graph G
  - h_t^{i,(0)} = x_t^i is the initial feature vector
  - MESSAGE, AGGREGATE, UPDATE are learnable functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MessageFunction(nn.Module):
    """MESSAGE^(l): transforms neighbor embeddings before aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(h))


class GNNLayer(nn.Module):
    """Single GNN message-passing layer.

    h_t^{i,(l)} = UPDATE(h_t^{i,(l-1)}, AGGREGATE({MESSAGE(h_t^{j,(l-1)}) | j in N(i)}))
    """

    def __init__(self, in_dim: int, out_dim: int, aggregation: str = "mean"):
        super().__init__()
        self.message_fn = MessageFunction(in_dim, out_dim)
        self.update_fn = nn.GRUCell(out_dim, in_dim)
        self.aggregation = aggregation

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, in_dim) node embeddings
            adj_matrix: (N, N) adjacency matrix of communication graph G
        Returns:
            updated_features: (N, in_dim)
        """
        N = node_features.size(0)
        messages = self.message_fn(node_features)  # (N, out_dim)

        adj = adj_matrix.float()
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)

        if self.aggregation == "mean":
            aggregated = torch.mm(adj, messages) / degree  # (N, out_dim)
        elif self.aggregation == "sum":
            aggregated = torch.mm(adj, messages)
        elif self.aggregation == "max":
            expanded_msg = messages.unsqueeze(0).expand(N, -1, -1)
            mask = adj.unsqueeze(-1).expand_as(expanded_msg)
            masked = expanded_msg * mask + (~mask.bool()).float() * (-1e9)
            aggregated = masked.max(dim=1)[0]
        else:
            aggregated = torch.mm(adj, messages) / degree

        updated = self.update_fn(aggregated, node_features)
        return updated


class GNNMessagePassing(nn.Module):
    """L-layer GNN for processing communication topology.

    Produces h_t^{i,(L)} — the final node embedding capturing
    agent i's state and its neighborhood context.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        aggregation: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, aggregation))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        observations: torch.Tensor,
        adj_matrix: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            observations: (N, obs_dim) raw local observations x_t^i
            adj_matrix: (N, N) communication graph adjacency
            return_all_layers: if True, return list of all layer outputs
        Returns:
            h_L: (N, hidden_dim) final embeddings h_t^{i,(L)}
        """
        h = F.relu(self.input_proj(observations))  # h_t^{i,(0)}

        all_layers = [h]
        for layer in self.layers:
            h = layer(h, adj_matrix)
            h = self.dropout(h)
            h = self.layer_norm(h)
            all_layers.append(h)

        if return_all_layers:
            return all_layers
        return h
