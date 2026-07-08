"""
Episode Replay Buffer for QMIX training.

Stores complete episodes of multi-agent interaction for experience replay.
Each episode contains sequences of (observations, actions, team rewards,
adjacency matrices, global states, valid-action masks, dones).

Sampling pads every episode to max_len + 1 (training_eval plan 3.2,
PyMARL-style filled slot) so the TD targets `[:, :-1]` cover every REAL
transition — including the terminal one of the longest episode in the batch.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

from .agent_network import NUM_ACTIONS


@dataclass
class EpisodeStep:
    observations: np.ndarray       # (N, obs_dim)
    actions: np.ndarray            # (N-1,) action indices
    team_reward: float             # shared scalar reward (reward v2)
    adj_matrix: np.ndarray         # (N, N) communication graph
    global_state: np.ndarray       # (state_dim,)
    done: bool = False
    token_usage: int = 0           # total tokens used this step
    # Valid-action mask (N-1, n_actions) active when the step was taken.
    # Consumed by train_step: invalid actions are excluded from the TD
    # target's argmax (training_eval plan 3.1). None = all actions valid.
    mask: Optional[np.ndarray] = None


@dataclass
class EpisodeBatch:
    """A batch of episodes for training."""
    obs: torch.Tensor              # (B, T, N, obs_dim)
    actions: torch.Tensor          # (B, T, N-1)
    avail_actions: torch.Tensor    # (B, T, N-1, n_actions) bool valid-action masks
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

        # Steps recorded without a mask count as all-valid (plan 3.1).
        n_acting = actions.shape[1]
        n_actions = next(
            (s.mask.shape[-1] for s in self.steps if s.mask is not None),
            NUM_ACTIONS,
        )
        avail = np.stack([
            np.asarray(s.mask, dtype=bool) if s.mask is not None
            else np.ones((n_acting, n_actions), dtype=bool)
            for s in self.steps
        ])

        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "avail": torch.tensor(avail, dtype=torch.bool),
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

        ep_tensors = [ep.to_tensors() for ep in episodes]
        n_nodes = ep_tensors[0]["obs"].shape[1]        # roster incl. Collector
        n_acting = ep_tensors[0]["actions"].shape[1]   # acting agents (N-1)
        n_actions = ep_tensors[0]["avail"].shape[2]
        obs_dim = ep_tensors[0]["obs"].shape[2]
        state_dim = ep_tensors[0]["global_state"].shape[1]

        # Pad to max_len + 1 (plan 3.2): the extra filled slot exists only to
        # be indexed by target_q_tot[:, 1:] for the terminal transition of
        # max-length episodes; its content is multiplied by (1 - done) = 0.
        T_pad = max_len + 1
        obs_batch = torch.zeros(B, T_pad, n_nodes, obs_dim)
        act_batch = torch.zeros(B, T_pad, n_acting, dtype=torch.long)
        # Padded slots keep all-valid masks: a fully-invalid row would make
        # the masked argmax -inf everywhere (plan 3.1).
        avail_batch = torch.ones(B, T_pad, n_acting, n_actions, dtype=torch.bool)
        rew_batch = torch.zeros(B, T_pad)
        adj_batch = torch.zeros(B, T_pad, n_nodes, n_nodes)
        state_batch = torch.zeros(B, T_pad, state_dim)
        mask_batch = torch.zeros(B, T_pad)
        done_batch = torch.zeros(B, T_pad)

        for i, tensors in enumerate(ep_tensors):
            T = tensors["obs"].shape[0]
            obs_batch[i, :T] = tensors["obs"]
            act_batch[i, :T] = tensors["actions"]
            avail_batch[i, :T] = tensors["avail"]
            rew_batch[i, :T] = tensors["rewards"]
            adj_batch[i, :T] = tensors["adj"]
            state_batch[i, :T] = tensors["global_state"]
            mask_batch[i, :T] = 1.0
            done_batch[i, :T] = tensors["done"]
            done_batch[i, T:] = 1.0  # done carried past the episode's end

        return EpisodeBatch(
            obs=obs_batch,
            actions=act_batch,
            avail_actions=avail_batch,
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
