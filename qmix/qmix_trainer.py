"""
QMIX Centralized Training (Section 4 from the paper).

Trains the entire system (GNNs, RNNs, mixing network) jointly by minimizing TD loss:
  L(θ) = E[(y^tot - Q_tot(τ, u; θ))^2]

where the target y^tot is:
  y^tot = R̄ + γ * max_{u'} Q_tot(τ', u'; θ^-)

Gradients flow through mixing network → MLPs → RNNs → GNNs of each agent.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from .agent_network import AgentQNetwork, NUM_ACTIONS
from .mixing_network import QMIXMixingNetwork
from .replay_buffer import ReplayBuffer, EpisodeBatch, Episode, EpisodeStep
from utils.log import get_logger

logger = get_logger("qmix_trainer")


class QMIXTrainer:
    """Complete QMIX training system.

    Networked MMDP formulation:
      (N, S, {A_i}, P, {R_i}, γ, G)

    Optimizes team-average return:
      J = E[Σ_{t=0}^∞ γ^t * R̄_t]
      where R̄_t = (1/N) * Σ_{i=1}^N R_i(s_t, u_t)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int = NUM_ACTIONS,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        rnn_hidden_dim: int = 128,
        mixing_hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        target_update_interval: int = 200,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        grad_clip: float = 10.0,
        token_penalty_weight: float = 0.1,
        accuracy_reward_weight: float = 1.0,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.token_penalty_weight = token_penalty_weight
        self.accuracy_reward_weight = accuracy_reward_weight
        self.device = device
        self.training_step = 0

        self.agent_network = AgentQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(device)

        self.mixing_network = QMIXMixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim,
        ).to(device)

        self.target_agent_network = copy.deepcopy(self.agent_network)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)
        self.target_agent_network.eval()
        self.target_mixing_network.eval()

        self.params = list(self.agent_network.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def compute_reward(
        self,
        accuracy: float,
        tokens_used: int,
        max_tokens: int = 10000,
    ) -> float:
        """Compute composite reward: accuracy reward - token penalty.

        Goal: maximize accuracy, minimize token inference usage.
        """
        accuracy_reward = accuracy * self.accuracy_reward_weight
        token_ratio = min(tokens_used / max_tokens, 1.0)
        token_penalty = token_ratio * self.token_penalty_weight
        return accuracy_reward - token_penalty

    def select_actions(
        self,
        observations: torch.Tensor,
        adj_matrix: torch.Tensor,
        hidden_states: torch.Tensor,
        epsilon: float = 0.1,
    ) -> tuple:
        """Epsilon-greedy action selection for all agents.

        During training: explore with ε probability.
        During deployment: purely decentralized greedy.
        """
        with torch.no_grad():
            q_values, new_hidden = self.agent_network(
                observations.to(self.device),
                adj_matrix.to(self.device),
                hidden_states.to(self.device),
            )

        actions = torch.zeros(self.n_agents, dtype=torch.long)
        for i in range(self.n_agents):
            if np.random.random() < epsilon:
                actions[i] = np.random.randint(0, self.n_actions)
            else:
                actions[i] = q_values[i].argmax().item()

        return actions, new_hidden.cpu()

    def train_step(self) -> Optional[Dict[str, float]]:
        """Single training step: sample batch and minimize TD loss.

        L(θ) = E[(y^tot - Q_tot(τ, u; θ))^2]
        y^tot = R̄ + γ * max_{u'} Q_tot(τ', u'; θ^-)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch: EpisodeBatch = self.replay_buffer.sample(self.batch_size)
        batch = self._to_device(batch)

        B, T, N, _ = batch.obs.shape

        # Forward pass through agent network for entire sequence
        hidden = self.agent_network.init_hidden(N).to(self.device)
        hidden = hidden.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)

        all_q_values = []
        for t in range(T):
            obs_t = batch.obs[:, t]  # (B, N, obs_dim)
            adj_t = batch.adj[:, t]  # (B, N, N)

            batch_q = []
            batch_h = []
            for b in range(B):
                q_vals, h = self.agent_network(obs_t[b], adj_t[b], hidden[b * N:(b + 1) * N])
                batch_q.append(q_vals)
                batch_h.append(h)

            q_stacked = torch.stack(batch_q)  # (B, N, n_actions)
            hidden = torch.cat(batch_h, dim=0)
            all_q_values.append(q_stacked)

        all_q_values = torch.stack(all_q_values, dim=1)  # (B, T, N, n_actions)

        # Chosen Q-values: gather by actions taken
        actions_expanded = batch.actions.unsqueeze(-1)  # (B, T, N, 1)
        chosen_q = all_q_values.gather(-1, actions_expanded).squeeze(-1)  # (B, T, N)

        # Mix chosen Q-values into Q_tot
        q_tot_list = []
        for t in range(T):
            q_tot = self.mixing_network(chosen_q[:, t], batch.global_state[:, t])
            q_tot_list.append(q_tot)
        q_tot = torch.stack(q_tot_list, dim=1)  # (B, T)

        # Target Q-values using target networks
        with torch.no_grad():
            target_hidden = self.target_agent_network.init_hidden(N).to(self.device)
            target_hidden = target_hidden.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)

            target_q_values = []
            for t in range(T):
                obs_t = batch.obs[:, t]
                adj_t = batch.adj[:, t]

                batch_tq = []
                batch_th = []
                for b in range(B):
                    tq, th = self.target_agent_network(obs_t[b], adj_t[b], target_hidden[b * N:(b + 1) * N])
                    batch_tq.append(tq)
                    batch_th.append(th)

                tq_stacked = torch.stack(batch_tq)
                target_hidden = torch.cat(batch_th, dim=0)
                target_q_values.append(tq_stacked)

            target_q_values = torch.stack(target_q_values, dim=1)  # (B, T, N, n_actions)
            target_max_q = target_q_values.max(dim=-1)[0]  # (B, T, N)

            target_q_tot_list = []
            for t in range(T):
                tq_tot = self.target_mixing_network(target_max_q[:, t], batch.global_state[:, t])
                target_q_tot_list.append(tq_tot)
            target_q_tot = torch.stack(target_q_tot_list, dim=1)  # (B, T)

        # TD targets: y^tot = R̄ + γ * max_{u'} Q_tot(τ', u'; θ^-)
        targets = batch.rewards[:, :-1] + self.gamma * (1 - batch.done[:, :-1]) * target_q_tot[:, 1:]

        # TD loss: L(θ) = E[(y^tot - Q_tot(τ, u; θ))^2]
        q_tot_current = q_tot[:, :-1]
        mask = batch.mask[:, :-1]

        td_error = (targets - q_tot_current) * mask
        loss = (td_error ** 2).sum() / mask.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_interval == 0:
            self._update_targets()

        return {
            "loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "td_error_abs": td_error.abs().mean().item(),
            "training_step": self.training_step,
        }

    def _update_targets(self):
        """Hard update target networks: θ^- ← θ"""
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        logger.info(f"Updated target networks at step {self.training_step}")

    def _to_device(self, batch: EpisodeBatch) -> EpisodeBatch:
        return EpisodeBatch(
            obs=batch.obs.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            adj=batch.adj.to(self.device),
            global_state=batch.global_state.to(self.device),
            mask=batch.mask.to(self.device),
            done=batch.done.to(self.device),
        )

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "agent_network": self.agent_network.state_dict(),
            "mixing_network": self.mixing_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)
        logger.info(f"Saved QMIX checkpoint to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_network.load_state_dict(checkpoint["agent_network"])
        self.mixing_network.load_state_dict(checkpoint["mixing_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
        self._update_targets()
        logger.info(f"Loaded QMIX checkpoint from {path} (step {self.training_step})")
