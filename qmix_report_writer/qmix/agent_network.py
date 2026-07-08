"""
Individual Agent Q-Network (Sections 2c, 2d from the paper).

Architecture per agent i:
  1. GNN processes observations through communication graph → h_t^{i,(L)}
  2. RNN (GRU) processes temporal history:
       z_t^i = RNN(z_{t-1}^i, h_t^{i,(L)}; θ_rnn)
  3. MLP computes Q-values for each action:
       Q_i(τ_t^i, ·; θ_i) = MLP(z_t^i; θ_mlp)

  where θ_i = {θ_gnn, θ_rnn, θ_mlp}
"""

import torch
import torch.nn as nn

from .gnn import GNNMessagePassing

# Agent actions controlling communication and execution strategy (v2 —
# upgrade-plan Stage 4.1). Changes from v1:
#   - solo_process became a true no_op: the agent is SKIPPED this round (no
#     LLM call, no retrieval) instead of executing without communicating
#     (report 2.3). "Think privately" (the old solo) can be re-added later if
#     an ablation wants it.
#   - terminate was dropped (decision D5): episodes end when the phase
#     pipeline completes, the handcrafted way.
# Per-phase/per-role validity of these actions is defined in action_masks.py.
ACTION_NAMES = [
    "no_op",                                     # 0: Skip execution this round entirely
    "broadcast_all",                             # 1: Send output to all other active acting agents
    *[f"selective_query{i}" for i in range(4)],  # 2 -> 5: Send output to acting agent i
    "aggregate_refine",                          # 6: Receive from all other active acting agents
    "append",                                    # 7: Send output to the Collector (report append)
]
NUM_ACTIONS = len(ACTION_NAMES)


class AgentQNetwork(nn.Module):
    """Individual agent Q-network with shared GNN + per-agent RNN + MLP.

    This network is shared across all agents (parameter sharing), but each
    agent maintains its own hidden state z_t^i in the RNN.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = NUM_ACTIONS,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        rnn_hidden_dim: int = 128,
        mlp_hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        self.gnn = GNNMessagePassing(
            obs_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
        )

        self.rnn = nn.GRUCell(gnn_hidden_dim, rnn_hidden_dim)

        self.q_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_actions),
        )

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize RNN hidden state z_0^i = 0."""
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        adj_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple:
        """
        Single-step forward pass for all agents (optionally batched, plan 3.5).

        Args:
            obs: (N, obs_dim) or (B, N, obs_dim) observations at time t
            adj_matrix: (N, N) or (B, N, N) communication graph
            hidden_state: (N, rnn_hidden_dim) or (B*N, rnn_hidden_dim)
                z_{t-1}^i for each agent (flat across the batch)
        Returns:
            q_values: (N, n_actions) or (B, N, n_actions)
            new_hidden: same 2D layout as hidden_state
        """
        gnn_out = self.gnn(obs, adj_matrix)  # h_t^{i,(L)}: (..., gnn_hidden_dim)

        flat = gnn_out.reshape(-1, gnn_out.shape[-1])
        new_hidden = self.rnn(flat, hidden_state)  # z_t^i: (B*N, rnn_hidden_dim)

        q_values = self.q_head(new_hidden)  # Q_i: (B*N, n_actions)
        if obs.dim() == 3:
            q_values = q_values.reshape(obs.shape[0], obs.shape[1], -1)

        return q_values, new_hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        adj_seq: torch.Tensor,
        hidden_init: torch.Tensor = None,
    ) -> tuple:
        """
        Process a full episode sequence.

        Args:
            obs_seq: (T, N, obs_dim) observations over time
            adj_seq: (T, N, N) adjacency matrices over time
            hidden_init: (N, rnn_hidden_dim) initial hidden state
        Returns:
            q_values_seq: (T, N, n_actions)
            hidden_states: (T, N, rnn_hidden_dim)
        """
        T, N, _ = obs_seq.shape
        if hidden_init is None:
            hidden_init = self.init_hidden(N).to(obs_seq.device)

        q_values_list = []
        hidden_list = []
        h = hidden_init

        for t in range(T):
            q_vals, h = self.forward(obs_seq[t], adj_seq[t], h)
            q_values_list.append(q_vals)
            hidden_list.append(h)

        return torch.stack(q_values_list), torch.stack(hidden_list)
