"""
GNN-based Q-Function Learning (Section 2 from the paper).

Implements the message-passing GNN for inter-agent communication:
  h_t^{i,(l)} = UPDATE^(l)(h_t^{i,(l-1)}, AGGREGATE^(l)({MESSAGE^(l)(h_t^{j,(l-1)}) | j in N(i)}))

where:
  - N(i) is the set of neighbors of agent i in communication graph G
  - h_t^{i,(0)} = x_t^i is the initial feature vector
  - MESSAGE, AGGREGATE, UPDATE are learnable functions

Aggregation direction (training_eval plan 3.6, decision OD-B): `build_adj`
stores A[i, j] = 1 ⇔ i SENDS to j, so a node must aggregate over its
IN-edges — the effective adjacency used here is Â = Aᵀ + I (transpose +
self-loops), row-normalized. The transpose lives HERE; build_adj semantics
are unchanged.

All forwards accept both single graphs ((N, d), (N, N)) and batches
((B, N, d), (B, N, N)) — train_step runs one batched forward per timestep
(plan 3.5).
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.message_fn = MessageFunction(in_dim, out_dim)
        self.update_fn = nn.GRUCell(out_dim, in_dim)
        self.aggregation = aggregation

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, in_dim) or (B, N, in_dim) node embeddings
            adj_matrix: (N, N) or (B, N, N) EFFECTIVE adjacency (row i = the
                nodes i aggregates FROM; GNNMessagePassing passes Â = Aᵀ + I)
        Returns:
            updated_features: same shape as node_features
        """
        single = node_features.dim() == 2
        h = node_features.unsqueeze(0) if single else node_features
        adj = adj_matrix.float()
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        B, N, _ = h.shape
        messages = self.message_fn(h)  # (B, N, out_dim)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)

        if self.aggregation == "mean":
            aggregated = torch.matmul(adj, messages) / degree  # (B, N, out_dim)
        elif self.aggregation == "sum":
            aggregated = torch.matmul(adj, messages)
        elif self.aggregation == "max":
            # expanded[b, i, j] = message of node j; invalid edges -> -1e9.
            expanded = messages.unsqueeze(1).expand(-1, N, -1, -1)
            edge_mask = adj.unsqueeze(-1).bool()
            masked = torch.where(edge_mask, expanded,
                                 torch.full_like(expanded, -1e9))
            aggregated = masked.max(dim=2)[0]
        else:
            aggregated = torch.matmul(adj, messages) / degree

        # GRUCell wants 2D input: flatten the batch/node dims together.
        updated = self.update_fn(
            aggregated.reshape(B * N, self.out_dim),
            h.reshape(B * N, self.in_dim),
        ).reshape(B, N, self.in_dim)
        return updated.squeeze(0) if single else updated


class GNNMessagePassing(nn.Module):
    """L-layer GNN for processing communication topology.

    Produces h_t^{i,(L)} — the final node embedding capturing
    agent i's state and its neighborhood context.

    No dropout (training_eval plan 3.4): a value network's action selection
    and TD targets must be deterministic; LayerNorm is kept.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, aggregation))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        observations: torch.Tensor,
        adj_matrix: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            observations: (N, obs_dim) or (B, N, obs_dim) raw observations
            adj_matrix: (N, N) or (B, N, N) communication graph in build_adj
                convention (A[i, j] = 1 ⇔ i sends to j)
            return_all_layers: if True, return list of all layer outputs
        Returns:
            h_L: (N, hidden_dim) or (B, N, hidden_dim) final embeddings
        """
        # In-edge aggregation + self-loops (plan 3.6, OD-B): who talks TO me.
        adj = adj_matrix.float()
        eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
        adj_eff = (adj.transpose(-1, -2) + eye).clamp(max=1.0)

        h = F.relu(self.input_proj(observations))  # h_t^{i,(0)}

        all_layers = [h]
        for layer in self.layers:
            h = layer(h, adj_eff)
            h = self.layer_norm(h)
            all_layers.append(h)

        if return_all_layers:
            return all_layers
        return h
