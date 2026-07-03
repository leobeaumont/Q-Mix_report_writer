"""
Observation builder tests (upgrade-plan Stage 5).

Offline (no Ollama): default config uses byte-hash text features, so nothing
here touches the embed endpoint. Covers:
  * obs_dim / state_dim consistency with the actual observation matrix,
  * the phase one-hot tracks PhaseState,
  * the 6 structured global scalars reflect ReportState,
  * the per-agent sentinel flag fires on a bracketed-sentinel last output,
  * the embedding path stays OFF by default (deterministic obs).

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_observations.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from qmix_report_writer.utils.globals import ReportState
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import PhaseType
from qmix_report_writer.qmix.observations import (
    build_observations, build_adj, build_global_state,
    get_obs_dim, get_state_dim, PHASE_ORDER, _obs_cfg,
    _structured_global_features,
)


class _FakeNode:
    def __init__(self, agent_name, outputs=None):
        self.agent_name = agent_name
        self.outputs = outputs or []
        self.spatial_predecessors = []
        self.spatial_successors = []
        self.token_usage = 0


def _reset():
    ReportState._instance = None
    PhaseState._instance = None


def _nodes(**out):
    """Roster of fake nodes; out maps agent_name -> last output string."""
    names = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
    return {
        f"{n}_{i}": _FakeNode(n, [out[n]] if n in out else [])
        for i, n in enumerate(names)
    }


def test_obs_dims_match_matrix():
    _reset()
    nodes = _nodes()
    obs = build_observations(nodes, "Some report task")
    assert obs.shape == (len(nodes), get_obs_dim()), \
        f"obs {obs.shape} vs ({len(nodes)}, {get_obs_dim()})"
    adj = build_adj(nodes)
    state = build_global_state(obs, adj)
    assert state.shape == (get_state_dim(len(nodes)),)
    # Default config: embeddings OFF (deterministic, offline).
    assert _obs_cfg().get("use_embeddings", False) is False
    # Same task -> identical observations (byte-hash determinism).
    obs2 = build_observations(_nodes(), "Some report task")
    assert np.allclose(obs, obs2)
    print("PASS  test_obs_dims_match_matrix")


def test_phase_one_hot_tracks_state():
    _reset()
    PhaseState.instance().set_phase(PhaseType.DRAFTING)
    obs = build_observations(_nodes(), "task")
    # Phase block sits after text(2*dim)+id(16)+extra(3).
    from qmix_report_writer.qmix.observations import _text_dim
    start = 2 * _text_dim() + 16 + 3
    phase_block = obs[0, start:start + len(PHASE_ORDER)]
    assert phase_block[PHASE_ORDER.index(PhaseType.DRAFTING)] == 1.0
    assert phase_block.sum() == 1.0
    print("PASS  test_phase_one_hot_tracks_state")


def test_structured_features_reflect_state():
    _reset()
    rs = ReportState.instance()
    rs.planned_sections = ["A", "B", "C", "D"]
    rs.append("## A\n\nBody one.", "progress")
    rs.append("## B\n\nBody two.", "progress")
    rs.task = "[SECTION_COMPLETE — ASSIGN NEXT SECTION]"

    nodes = _nodes(Researcher="[RESEARCH_EXHAUSTED] nothing found")
    feats = _structured_global_features(nodes)
    written, remaining, length_ratio, complete, skipped, exhausted = feats

    assert written == 2 / 10.0
    assert remaining == (4 - 2) / 10.0
    assert length_ratio > 0.0
    assert complete == 1.0 and skipped == 0.0
    assert exhausted == 1.0

    # Skipped sentinel flips the two append flags.
    rs.task = "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
    feats = _structured_global_features(_nodes())
    assert feats[3] == 0.0 and feats[4] == 1.0
    assert feats[5] == 0.0  # no researcher exhaustion this time
    print("PASS  test_structured_features_reflect_state")


def test_per_agent_sentinel_flag():
    _reset()
    nodes = _nodes(Researcher="[HOLD]", LeadArchitect="Real strategic prose here.")
    obs = build_observations(nodes, "task")
    node_names = [n.agent_name for n in nodes.values()]
    r_idx = node_names.index("Researcher")
    la_idx = node_names.index("LeadArchitect")
    # Sentinel flag is the final feature.
    assert obs[r_idx, -1] == 1.0, "sentinel output should set the flag"
    assert obs[la_idx, -1] == 0.0, "prose output should not set the flag"
    print("PASS  test_per_agent_sentinel_flag")


def _run_all():
    passed = failed = 0
    for name, fn in [
        ("test_obs_dims_match_matrix", test_obs_dims_match_matrix),
        ("test_phase_one_hot_tracks_state", test_phase_one_hot_tracks_state),
        ("test_structured_features_reflect_state", test_structured_features_reflect_state),
        ("test_per_agent_sentinel_flag", test_per_agent_sentinel_flag),
    ]:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
