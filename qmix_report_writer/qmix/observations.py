"""
Observation / global-state builders for the QMIX controller.

History:
  * Stage 4.2 — ported the legacy byte-hash features + added a phase one-hot.
  * Stage 5.1 — added structured features read from ReportState (report 2.7):
      6 global scalars shared by every agent + 1 per-agent sentinel flag.
  * Stage 5.2 — the two text slots (task, progress summary) are byte-hashes by
      default; `qmix.obs.use_embeddings: true` swaps them for nomic-embed-text
      embeddings (same slot width, so obs_dim is unchanged unless
      `qmix.obs.embedding_dim` is also raised). Default OFF → deterministic,
      no Ollama dependency, offline tests unaffected.

Per-agent observation (obs_dim, default = 63):
    T   task text features        (byte-hash or nomic embedding; T=embedding_dim, default 16)
    T   progress text features     (same)
    16  agent-identity one-hot
     3  [has_output, n_neighbors_ratio, token_usage/10000]
     5  phase one-hot (PLANNING, RESEARCH, DRAFTING, SECTION_REVIEW, VALIDATION)
     6  structured global scalars  (see _structured_global_features)
     1  last-output-is-sentinel flag (per agent)

Global state = concat of all agent observations + [edge_count, density, n_agents].
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Dict, List

import numpy as np

from qmix_report_writer.handcrafted_graph.phases import PhaseType
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.utils.config import get_config
from qmix_report_writer.utils.globals import ReportState

logger = logging.getLogger("qmix.observations")

_ID_DIM = 16
_EXTRA_DIM = 3
_STRUCTURED_DIM = 6
_SENTINEL_DIM = 1
PHASE_ORDER: List[PhaseType] = [
    PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING,
    PhaseType.SECTION_REVIEW, PhaseType.VALIDATION,
]

# A line led by an all-caps bracketed token, e.g. "[RESEARCH_EXHAUSTED]",
# "[HOLD]", "[NO NEW EVIDENCE]". Used for the per-agent sentinel flag.
_BRACKET_SENTINEL_RE = re.compile(r'^\s*\[[A-Z0-9_ ]{3,}\]')

# One embedding cache per process; keyed by text. Cleared implicitly by the
# task/progress strings changing. Small (2 live keys in practice).
_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _obs_cfg() -> dict:
    return (get_config().get("qmix", {}) or {}).get("obs", {}) or {}


def _text_dim() -> int:
    return int(_obs_cfg().get("embedding_dim", 16))


def get_obs_dim() -> int:
    t = _text_dim()
    return t + t + _ID_DIM + _EXTRA_DIM + len(PHASE_ORDER) + _STRUCTURED_DIM + _SENTINEL_DIM


def get_state_dim(n_agents: int) -> int:
    graph_stats = 3  # edge_count, density, n_agents
    return n_agents * get_obs_dim() + graph_stats


# ------------------------------------------------------------------
# Text features (byte-hash default; nomic embedding when enabled)
# ------------------------------------------------------------------

def _hash_features(text: str, dim: int) -> np.ndarray:
    """Simple byte-hash feature extraction (legacy v1 features)."""
    features = np.zeros(dim)
    for i, c in enumerate(text.encode()[:dim * 4]):
        features[i % dim] += c / 255.0
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def _ollama_base_url() -> str:
    llm = get_config().get("llm", {}) or {}
    return (
        llm.get("providers", {}).get("ollama", {}).get("base_url")
        or "http://localhost:11434"
    )


def _embed_features(text: str, dim: int) -> np.ndarray:
    """nomic-embed-text embedding, truncated/padded to `dim` and L2-normed.

    Cached per text. Fails soft to the byte-hash so a missing embed endpoint
    never crashes observation building.
    """
    if not text:
        return np.zeros(dim)
    cached = _EMBED_CACHE.get(text)
    if cached is not None and len(cached) == dim:
        return cached
    try:
        payload = json.dumps({"model": "nomic-embed-text", "input": [text]}).encode()
        req = urllib.request.Request(
            f"{_ollama_base_url()}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            vec = np.asarray(json.loads(resp.read())["embeddings"][0], dtype=float)
    except Exception as exc:
        logger.warning(f"Embedding call failed ({exc}); falling back to byte-hash.")
        return _hash_features(text, dim)
    # Truncate or zero-pad to the configured slot width, then L2-normalize.
    if len(vec) >= dim:
        vec = vec[:dim]
    else:
        vec = np.concatenate([vec, np.zeros(dim - len(vec))])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    _EMBED_CACHE[text] = vec
    return vec


def _text_features(text: str, dim: int, use_embeddings: bool) -> np.ndarray:
    return _embed_features(text, dim) if use_embeddings else _hash_features(text, dim)


# ------------------------------------------------------------------
# Structured features
# ------------------------------------------------------------------

def _phase_one_hot() -> np.ndarray:
    one_hot = np.zeros(len(PHASE_ORDER))
    phase = PhaseState.instance().current_phase
    if phase in PHASE_ORDER:
        one_hot[PHASE_ORDER.index(phase)] = 1.0
    return one_hot


def _length_goal() -> int:
    return int((get_config().get("reward", {}) or {}).get("length_goal", 25000))


def _structured_global_features(nodes: Dict[str, object]) -> np.ndarray:
    """Six ReportState-derived scalars, shared by every agent this round.

    [ written/10, remaining_planned/10, length_ratio(capped 2),
      last_append_complete, last_append_skipped, research_exhausted ]
    """
    rs = ReportState.instance()
    n_written = len(rs.sections)
    remaining = max(0, len(rs.planned_sections) - n_written)
    length_ratio = min(len(rs.content) / max(_length_goal(), 1), 2.0)

    task = str(rs.task or "")
    append_complete = 1.0 if "SECTION_COMPLETE" in task else 0.0
    append_skipped = 1.0 if "SECTION_SKIPPED" in task else 0.0

    research_exhausted = 0.0
    for node in nodes.values():
        if getattr(node, "agent_name", "") == "Researcher" and node.outputs:
            if "[RESEARCH_EXHAUSTED]" in str(node.outputs[-1] or ""):
                research_exhausted = 1.0
            break

    return np.array([
        n_written / 10.0,
        remaining / 10.0,
        length_ratio,
        append_complete,
        append_skipped,
        research_exhausted,
    ])


def _last_output_is_sentinel(node) -> float:
    if not node.outputs:
        return 0.0
    last = str(node.outputs[-1] or "").strip()
    return 1.0 if _BRACKET_SENTINEL_RE.match(last) else 0.0


# ------------------------------------------------------------------
# Public builders
# ------------------------------------------------------------------

def build_observations(nodes: Dict[str, object], task: str) -> np.ndarray:
    """Observation matrix (n_agents, obs_dim) over ALL nodes (Collector incl.).

    Node order = dict insertion order = roster order, matching the legacy
    convention (acting agents first, Collector last).
    """
    dim = _text_dim()
    use_emb = bool(_obs_cfg().get("use_embeddings", False))

    task_features = _text_features(task, dim, use_emb)
    state_features = _text_features(ReportState.instance().progress, dim, use_emb)
    phase_features = _phase_one_hot()
    structured = _structured_global_features(nodes)
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
            structured,
            [_last_output_is_sentinel(node)],
        ]))

    return np.stack(obs_list)


def build_adj(nodes: Dict[str, object]) -> np.ndarray:
    """Adjacency matrix of the current spatial connections.

    Convention: A[i, j] = 1 ⇔ i SENDS to j. The GNN aggregates over
    IN-edges, so it internally uses Â = Aᵀ + I (training_eval plan 3.6,
    OD-B) — the transpose lives in GNNMessagePassing, not here.
    """
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
