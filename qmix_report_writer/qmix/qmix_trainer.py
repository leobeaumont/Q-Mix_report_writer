"""
QMIX Centralized Training (Section 4 from the paper).

Trains the entire system (GNNs, RNNs, mixing network) jointly by minimizing TD loss:
  L(θ) = E[(y^tot - Q_tot(τ, u; θ))^2]

where the target y^tot uses Double-DQN (training_eval plan 3.3): the online
network picks the (mask-valid) argmax action, the target network evaluates it:
  u'* = argmax_{u' valid} Q(τ', u'; θ)
  y^tot = R̄ + γ * (1 - done) * Q_tot(τ', u'*; θ^-)

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
from .replay_buffer import ReplayBuffer, EpisodeBatch
from qmix_report_writer.utils.log import get_logger

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
        device: str = "cpu",
    ):
        self.n_agents = n_agents  # with collector agent
        self.n_acting_agents = n_agents - 1  # without collector agent
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.grad_clip = grad_clip
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
            n_agents=self.n_acting_agents,  # collector doesn't require Q-value or action selection
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

    # NOTE: reward composition moved to qmix_report_writer/evaluation/reward.py
    # (training_eval plan 2.5, report B0.4) — the trainer is pure TD machinery.

    def select_actions(
        self,
        observations: torch.Tensor,
        adj_matrix: torch.Tensor,
        hidden_states: torch.Tensor,
        epsilon: float = 0.1,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Epsilon-greedy action selection for all agents.

        During training: explore with ε probability.
        During deployment: purely decentralized greedy.

        mask: optional (n_acting, n_actions) bool tensor of valid actions.
        Masked entries are excluded from BOTH the greedy argmax and the
        ε-random draw (upgrade-plan Stage 4.3). A fully-masked row falls back
        to the unmasked distribution (defensive; the mask builder never emits
        one).
        """
        # eval()/restore around inference (plan 3.4): no stochastic layer
        # remains today, but action selection must stay deterministic even if
        # one is ever added.
        was_training = self.agent_network.training
        self.agent_network.eval()
        try:
            with torch.no_grad():
                q_values, new_hidden = self.agent_network(
                    observations.to(self.device),
                    adj_matrix.to(self.device),
                    hidden_states.to(self.device),
                )
        finally:
            if was_training:
                self.agent_network.train()

        actions = torch.zeros(self.n_acting_agents, dtype=torch.long)
        for i in range(self.n_acting_agents):
            valid = None
            if mask is not None:
                valid = torch.nonzero(mask[i]).flatten()
                if len(valid) == 0:
                    valid = None
            if np.random.random() < epsilon:
                if valid is not None:
                    actions[i] = valid[np.random.randint(len(valid))].item()
                else:
                    actions[i] = np.random.randint(0, self.n_actions)
            else:
                if valid is not None:
                    q = q_values[i].clone()
                    q[~mask[i].to(q.device)] = float("-inf")
                    actions[i] = int(q.argmax().item())
                else:
                    actions[i] = q_values[i].argmax().item()

        return actions, new_hidden.cpu()

    def train_step(self) -> Optional[Dict[str, float]]:
        """Single training step: sample batch and minimize TD loss.

        L(θ) = E[(y^tot - Q_tot(τ, u; θ))^2]
        y^tot = R̄ + γ * (1 - done) * Q_tot(τ', argmax_{u' valid} Q(τ', u'; θ); θ^-)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch: EpisodeBatch = self.replay_buffer.sample(self.batch_size)
        batch = self._to_device(batch)

        B, T, N, _ = batch.obs.shape

        # Forward pass through agent network for the entire sequence — one
        # BATCHED forward per timestep (plan 3.5); the GRU time loop stays.
        hidden = self.agent_network.init_hidden(N).to(self.device)
        hidden = hidden.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)

        all_q_values = []
        for t in range(T):
            q_t, hidden = self.agent_network(
                batch.obs[:, t], batch.adj[:, t], hidden,
            )  # (B, N, n_actions), (B*N, rnn_hidden)
            all_q_values.append(q_t)

        all_q_values = torch.stack(all_q_values, dim=1)  # (B, T, N, n_actions)
        acting_q_values = all_q_values[:, :, :self.n_acting_agents, :]  # (B, T, N-1, n_actions) removed collector agent Q-value

        # Chosen Q-values: gather by actions taken
        actions_expanded = batch.actions.unsqueeze(-1)  # (B, T, N-1, 1)
        chosen_q = acting_q_values.gather(-1, actions_expanded).squeeze(-1)  # (B, T, N-1)

        # Mix chosen Q-values into Q_tot
        q_tot_list = []
        for t in range(T):
            q_tot = self.mixing_network(chosen_q[:, t], batch.global_state[:, t])
            q_tot_list.append(q_tot)
        q_tot = torch.stack(q_tot_list, dim=1)  # (B, T)

        # Target Q-values using target networks (batched, plan 3.5)
        with torch.no_grad():
            target_hidden = self.target_agent_network.init_hidden(N).to(self.device)
            target_hidden = target_hidden.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)

            target_q_values = []
            for t in range(T):
                tq_t, target_hidden = self.target_agent_network(
                    batch.obs[:, t], batch.adj[:, t], target_hidden,
                )
                target_q_values.append(tq_t)

            target_q_values = torch.stack(target_q_values, dim=1)  # (B, T, N, n_actions)
            target_acting_q = target_q_values[:, :, :self.n_acting_agents, :]  # (B, T, N-1, n_actions)

            # Double-DQN (plan 3.3): the ONLINE net picks a*, the TARGET net
            # evaluates it — replaces the target-side max. Invalid actions are
            # excluded from the argmax via the stored masks (plan 3.1); padded
            # slots carry all-valid masks, so no row is -inf everywhere, and
            # their values are killed by (1 - done) in the TD targets.
            online_q = acting_q_values.detach().clone()
            online_q[~batch.avail_actions] = float("-inf")
            best_actions = online_q.argmax(dim=-1, keepdim=True)  # (B, T, N-1, 1)
            target_chosen_q = target_acting_q.gather(-1, best_actions).squeeze(-1)  # (B, T, N-1)

            target_q_tot_list = []
            for t in range(T):
                tq_tot = self.target_mixing_network(target_chosen_q[:, t], batch.global_state[:, t])
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
            avail_actions=batch.avail_actions.to(self.device),
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

