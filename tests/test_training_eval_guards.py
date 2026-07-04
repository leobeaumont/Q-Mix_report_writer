"""
Guard tests for training_eval_upgrade_plan.md — regression pins written BEFORE
the changes. Green on pre-change code; must STAY green after every stage.

They pin the controller-seam ENVIRONMENT semantics the rework builds on (and
must not drift), not the scorer/trainer internals being replaced:

  * reward events fire only on real report growth (never on the action);
  * on_run_end flushes cleanly, marks done, and is idempotent;
  * action -> RoundPlan translation invariants;
  * EpisodeStep.mask is recorded with the right shape (Stage 3.1 consumes it);
  * the length-gaussian values (the math survives its Stage-2 move);
  * trainer checkpoint save/load round-trip.

Scheduled updates (documented in the plan's relaxed-constraint note):
  * Stage 2 moves the length gaussian into qmix_report_writer/evaluation —
    test_length_gaussian_values already prefers the new home when importable.
  * Stage 2.5 may rename controller internals (_reward_event) — update the
    patch target in test_reward_event_on_append_only, keep the assertions.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_training_eval_guards.py
"""

import asyncio
import dataclasses
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")

from qmix_report_writer.handcrafted_graph.phases import PhaseType
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
from qmix_report_writer.qmix.observations import get_obs_dim, get_state_dim
from qmix_report_writer.qmix.qmix_controller import QMIXRoundController, plan_from_actions
from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
from qmix_report_writer.qmix.replay_buffer import EpisodeStep
from qmix_report_writer.utils.globals import ReportState, Score, LengthGoal, SourceBuffer

ROSTER = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
ACTING = ROSTER[:-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_state():
    ReportState.instance().reset()
    PhaseState.instance().reset()
    SourceBuffer.instance().reset()
    try:
        Score.instance().reset()
        LengthGoal.instance().reset()
    except Exception:
        pass


def _stub_trainer():
    """Trainer stand-in for controller paths that never select actions."""
    return SimpleNamespace(n_actions=NUM_ACTIONS, compute_reward=lambda *a, **k: 0.5)


def _mk_step(**overrides):
    """Build an EpisodeStep passing only the fields the dataclass requires.

    Field-introspective so the guard survives cosmetic field removals
    (plan 3.7 drops the dead per-agent rewards array).
    """
    n, n_act = len(ROSTER), len(ACTING)
    defaults = {
        "observations": np.zeros((n, 4), dtype=np.float32),
        "actions": np.zeros(n_act, dtype=np.int64),
        "rewards": np.zeros(n_act, dtype=np.float32),
        "team_reward": 0.0,
        "adj_matrix": np.zeros((n, n), dtype=np.float32),
        "global_state": np.zeros(7, dtype=np.float32),
        "done": False,
    }
    defaults.update(overrides)
    names = {f.name for f in dataclasses.fields(EpisodeStep)}
    return EpisodeStep(**{k: v for k, v in defaults.items() if k in names})


class _FakeNode:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.outputs = []
        self.spatial_predecessors = []
        self.spatial_successors = []
        self.token_usage = 0


def _mk_controller(train=True, **kwargs):
    return QMIXRoundController(_stub_trainer(), ROSTER, train=train, **kwargs)


# ---------------------------------------------------------------------------
# Pin 1 — reward events fire only on real report growth
# ---------------------------------------------------------------------------

def test_reward_event_on_append_only():
    _reset_state()
    ctrl = _mk_controller(train=True)

    events = []

    async def _fake_event():
        events.append(1)

    # Stage 2.5 note: if _reward_event is renamed, re-point this patch only.
    ctrl._reward_event = _fake_event

    async def _drive():
        await ctrl.on_round_end(PhaseType.PLANNING, 0)      # no growth
        ReportState.instance().append("## S1\n\nProse.", "progress")
        await ctrl.on_round_end(PhaseType.DRAFTING, 1)      # growth -> event
        await ctrl.on_round_end(PhaseType.DRAFTING, 2)      # no growth again

    asyncio.run(_drive())
    assert events == [1], f"expected exactly one reward event, got {len(events)}"
    print("PASS  test_reward_event_on_append_only")


# ---------------------------------------------------------------------------
# Pin 2 — on_run_end: clean flush, done flag, idempotent
# ---------------------------------------------------------------------------

def test_on_run_end_idempotent_and_done():
    _reset_state()
    ctrl = _mk_controller(train=True)
    ctrl._step_buffer.extend([_mk_step(), _mk_step()])

    asyncio.run(ctrl.on_run_end())
    assert ctrl.episode.length == 2
    assert ctrl.episode.steps[-1].done is True
    assert ctrl._step_buffer == []
    assert all(np.isfinite(s.team_reward) for s in ctrl.episode.steps)

    asyncio.run(ctrl.on_run_end())  # second call must be a no-op
    assert ctrl.episode.length == 2
    print("PASS  test_on_run_end_idempotent_and_done")


# ---------------------------------------------------------------------------
# Pin 3 — action -> RoundPlan translation invariants
# ---------------------------------------------------------------------------

def test_plan_from_actions_invariants():
    topo = SimpleNamespace(required_agents=(), edges=())

    # no_op excludes; selective at an inactive target drops the edge;
    # broadcast/aggregate only touch active acting agents.
    plan = plan_from_actions(
        {"LeadArchitect": 0, "Researcher": 2, "DataAnalyst": 1, "Reviewer": 6},
        ACTING, topo,
    )
    assert "LeadArchitect" not in plan.active_agents
    assert ("Researcher", "LeadArchitect") not in plan.edges  # target no_op'd
    assert ("DataAnalyst", "Researcher") in plan.edges        # broadcast
    assert ("DataAnalyst", "LeadArchitect") not in plan.edges
    assert ("DataAnalyst", "Reviewer") in plan.edges
    assert ("Researcher", "Reviewer") in plan.edges           # aggregate_refine

    # Self-targeting selective query drops its edge (Researcher -> index 1 = itself).
    plan = plan_from_actions(
        {"LeadArchitect": 1, "Researcher": 3, "DataAnalyst": 0, "Reviewer": 0},
        ACTING, topo,
    )
    assert ("Researcher", "Researcher") not in plan.edges

    # Environment invariant: a Collector-requiring round keeps table edges into it.
    topo = SimpleNamespace(
        required_agents={"Collector"}, edges=[("DataAnalyst", "Collector")],
    )
    plan = plan_from_actions(
        {"LeadArchitect": 0, "Researcher": 0, "DataAnalyst": 1, "Reviewer": 0},
        ACTING, topo,
    )
    assert "Collector" in plan.active_agents
    assert ("DataAnalyst", "Collector") in plan.edges
    print("PASS  test_plan_from_actions_invariants")


# ---------------------------------------------------------------------------
# Pin 4 — round_plan records the action mask on the step (Stage 3.1 consumes it)
# ---------------------------------------------------------------------------

def test_episode_step_mask_recorded():
    _reset_state()
    trainer = QMIXTrainer(
        n_agents=len(ROSTER), obs_dim=get_obs_dim(),
        state_dim=get_state_dim(len(ROSTER)),
        gnn_hidden_dim=8, gnn_layers=1, rnn_hidden_dim=8, mixing_hidden_dim=4,
    )
    ctrl = QMIXRoundController(trainer, ROSTER, train=True, epsilon=0.0)
    nodes = {name: _FakeNode(name) for name in ROSTER}
    topo = SimpleNamespace(required_agents=(), edges=())

    async def _drive():
        await ctrl.round_plan(PhaseType.PLANNING, 0, topo, nodes, {"task": "T"})
        await ctrl.on_round_end(PhaseType.PLANNING, 0)

    asyncio.run(_drive())
    assert len(ctrl._step_buffer) == 1
    step = ctrl._step_buffer[0]
    assert step.mask is not None and step.mask.shape == (len(ACTING), NUM_ACTIONS)
    assert step.mask.dtype == bool
    for i in range(len(ACTING)):
        assert not step.mask[i][2 + i], "self-query must be masked"
    assert step.observations.shape == (len(ROSTER), get_obs_dim())
    print("PASS  test_episode_step_mask_recorded")


# ---------------------------------------------------------------------------
# Pin 5 — length-gaussian values (math survives its Stage-2 move)
# ---------------------------------------------------------------------------

def test_length_gaussian_values():
    _reset_state()
    try:
        # Stage 2.4 home
        from qmix_report_writer.evaluation.reward import length_gaussian
        assert abs(length_gaussian(25000, 25000, 8500) - 1.0) < 1e-9
        assert length_gaussian(0, 25000, 8500) < 0.05
    except ImportError:
        # Pre-change home: controller-side gaussian
        ctrl = _mk_controller(train=False, length_goal=25000, length_sigma=8500)
        ReportState.instance().content = "x" * 25000
        assert abs(ctrl._length_score() - 1.0) < 1e-9
        ReportState.instance().content = ""
        assert ctrl._length_score() < 0.05
    print("PASS  test_length_gaussian_values")


# ---------------------------------------------------------------------------
# Pin 6 — trainer checkpoint save/load round-trip
# ---------------------------------------------------------------------------

def test_trainer_checkpoint_roundtrip():
    kwargs = dict(
        n_agents=3, obs_dim=6, state_dim=21,
        gnn_hidden_dim=8, gnn_layers=1, rnn_hidden_dim=8, mixing_hidden_dim=4,
    )
    t1 = QMIXTrainer(**kwargs)
    t1.training_step = 7
    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    t1.save(path)

    t2 = QMIXTrainer(**kwargs)
    t2.load(path)
    assert t2.training_step == 7
    for k, v in t1.agent_network.state_dict().items():
        assert torch.equal(v, t2.agent_network.state_dict()[k])
    print("PASS  test_trainer_checkpoint_roundtrip")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    passed = failed = 0
    cases = [
        ("test_reward_event_on_append_only", test_reward_event_on_append_only),
        ("test_on_run_end_idempotent_and_done", test_on_run_end_idempotent_and_done),
        ("test_plan_from_actions_invariants", test_plan_from_actions_invariants),
        ("test_episode_step_mask_recorded", test_episode_step_mask_recorded),
        ("test_length_gaussian_values", test_length_gaussian_values),
        ("test_trainer_checkpoint_roundtrip", test_trainer_checkpoint_roundtrip),
    ]
    for name, fn in cases:
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
