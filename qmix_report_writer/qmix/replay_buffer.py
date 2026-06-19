"""
Episode Replay Buffer for QMIX training.

Stores complete episodes of multi-agent interaction for experience replay.
Each episode contains sequences of (observations, actions, rewards,
adjacency matrices, global states, dones).
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque


@dataclass
class EpisodeStep:
    observations: np.ndarray       # (N, obs_dim)
    actions: np.ndarray            # (N-1,) action indices
    rewards: np.ndarray            # (N-1,) per-agent rewards
    team_reward: float             # R̄ = (1/N) * Σ R_i
    adj_matrix: np.ndarray         # (N, N) communication graph
    global_state: np.ndarray       # (state_dim,)
    done: bool = False
    token_usage: int = 0           # total tokens used this step


@dataclass
class EpisodeBatch:
    """A batch of episodes for training."""
    obs: torch.Tensor              # (B, T, N, obs_dim)
    actions: torch.Tensor          # (B, T, N-1)
    rewards: torch.Tensor          # (B, T)     team rewards
    adj: torch.Tensor              # (B, T, N, N)
    global_state: torch.Tensor     # (B, T, state_dim)
    mask: torch.Tensor             # (B, T)     valid timestep mask
    done: torch.Tensor             # (B, T)


class Episode:
    """Single episode of multi-agent interaction."""

    def __init__(self):
        self.steps: List[EpisodeStep] = []

    def add_step(self, step: EpisodeStep):
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return sum(s.team_reward for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_usage for s in self.steps)

    def to_tensors(self):
        obs = np.stack([s.observations for s in self.steps])
        actions = np.stack([s.actions for s in self.steps])
        rewards = np.array([s.team_reward for s in self.steps])
        adj = np.stack([s.adj_matrix for s in self.steps])
        states = np.stack([s.global_state for s in self.steps])
        dones = np.array([s.done for s in self.steps], dtype=np.float32)

        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "adj": torch.tensor(adj, dtype=torch.float32),
            "global_state": torch.tensor(states, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
        }


class ReplayBuffer:
    """Fixed-size replay buffer storing complete episodes."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, episode: Episode):
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> EpisodeBatch:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        episodes = [self.buffer[i] for i in indices]

        max_len = max(ep.length for ep in episodes)
        B = len(episodes)

        # Pad episodes to same length
        ep_tensors = [ep.to_tensors() for ep in episodes]
        N_nodes = ep_tensors[0]["obs"].shape[1]  # 6 agents
        N_actions = ep_tensors[0]["actions"].shape[1]  # 5 actions
        obs_dim = ep_tensors[0]["obs"].shape[2]
        state_dim = ep_tensors[0]["global_state"].shape[1]

        obs_batch = torch.zeros(B, max_len, N_nodes, obs_dim)
        act_batch = torch.zeros(B, max_len, N_actions, dtype=torch.long)
        rew_batch = torch.zeros(B, max_len)
        adj_batch = torch.zeros(B, max_len, N_nodes, N_nodes)
        state_batch = torch.zeros(B, max_len, state_dim)
        mask_batch = torch.zeros(B, max_len)
        done_batch = torch.zeros(B, max_len)

        for i, tensors in enumerate(ep_tensors):
            T = tensors["obs"].shape[0]
            obs_batch[i, :T] = tensors["obs"]
            act_batch[i, :T] = tensors["actions"]
            rew_batch[i, :T] = tensors["rewards"]
            adj_batch[i, :T] = tensors["adj"]
            state_batch[i, :T] = tensors["global_state"]
            mask_batch[i, :T] = 1.0
            done_batch[i, :T] = tensors["done"]

        return EpisodeBatch(
            obs=obs_batch,
            actions=act_batch,
            rewards=rew_batch,
            adj=adj_batch,
            global_state=state_batch,
            mask=mask_batch,
            done=done_batch,
        )

    def __len__(self):
        return len(self.buffer)

    @property
    def avg_reward(self) -> float:
        if not self.buffer:
            return 0.0
        return np.mean([ep.total_reward for ep in self.buffer])

    @property
    def avg_tokens(self) -> float:
        if not self.buffer:
            return 0.0
        return np.mean([ep.total_tokens for ep in self.buffer])
