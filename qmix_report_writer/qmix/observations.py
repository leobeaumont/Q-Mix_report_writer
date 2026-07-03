"""
Observation / global-state builders for the QMIX controller (Stage 4.2).

Ported from the deleted legacy QMIXGraph (git: qmix_report_writer/graph/
graph.py, get_observation_features / get_adj_matrix / get_global_state) and
extended with a phase one-hot (report 2.7 lands the structured-feature
upgrade in Stage 5; this is the v1 feature set + phase).

Per-agent observation (obs_dim = 56):
    16  task byte-hash features
    16  progress-summary byte-hash features
    16  agent-identity one-hot (up to 16 agents)
     3  [has_output, n_neighbors_ratio, token_usage/10000]
     5  phase one-hot (PLANNING, RESEARCH, DRAFTING, SECTION_REVIEW, VALIDATION)

Global state = concat of all agent observations + [edge_count, density, n_agents].
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from qmix_report_writer.handcrafted_graph.phases import PhaseType
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.utils.globals import ReportState

_HASH_DIM = 16
_ID_DIM = 16
_EXTRA_DIM = 3
PHASE_ORDER: List[PhaseType] = [
    PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING,
    PhaseType.SECTION_REVIEW, PhaseType.VALIDATION,
]


def get_obs_dim() -> int:
    return _HASH_DIM + _HASH_DIM + _ID_DIM + _EXTRA_DIM + len(PHASE_ORDER)


def get_state_dim(n_agents: int) -> int:
    graph_stats = 3  # edge_count, density, n_agents
    return n_agents * get_obs_dim() + graph_stats


def _hash_features(text: str, dim: int = _HASH_DIM) -> np.ndarray:
    """Simple byte-hash feature extraction (legacy v1 features)."""
    features = np.zeros(dim)
    for i, c in enumerate(text.encode()[:dim * 4]):
        features[i % dim] += c / 255.0
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def _phase_one_hot() -> np.ndarray:
    one_hot = np.zeros(len(PHASE_ORDER))
    phase = PhaseState.instance().current_phase
    if phase in PHASE_ORDER:
        one_hot[PHASE_ORDER.index(phase)] = 1.0
    return one_hot


def build_observations(nodes: Dict[str, object], task: str) -> np.ndarray:
    """Observation matrix (n_agents, obs_dim) over ALL nodes (Collector incl.).

    Node order = dict insertion order = roster order, matching the legacy
    convention (acting agents first, Collector last).
    """
    task_features = _hash_features(task)
    state_features = _hash_features(ReportState.instance().progress)
    phase_features = _phase_one_hot()
    n_agents = len(nodes)

    obs_list = []
    for i, node in enumerate(nodes.values()):
        agent_feature = np.zeros(_ID_DIM)
        agent_feature[i % _ID_DIM] = 1.0

        has_output = 1.0 if node.outputs else 0.0
        n_neighbors = len(node.spatial_predecessors) + len(node.spatial_successors)

        obs_list.append(np.concatenate([
            task_features,
            state_features,
            agent_feature,
            [has_output, n_neighbors / max(n_agents, 1), node.token_usage / 10000.0],
            phase_features,
        ]))

    return np.stack(obs_list)


def build_adj(nodes: Dict[str, object]) -> np.ndarray:
    """Adjacency matrix of the current spatial connections."""
    node_list = list(nodes.values())
    n = len(node_list)
    matrix = np.zeros((n, n))
    for i, a in enumerate(node_list):
        for j, b in enumerate(node_list):
            if b in a.spatial_successors:
                matrix[i, j] = 1
    return matrix


def build_global_state(obs: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Global state for the mixing network: all observations + graph stats."""
    n_agents = obs.shape[0]
    graph_stats = np.array([
        adj.sum(),
        adj.sum() / max(n_agents ** 2, 1),
        n_agents,
    ])
    return np.concatenate([obs.flatten(), graph_stats])
