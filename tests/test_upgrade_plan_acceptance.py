"""
Acceptance tests for qmix_upgrade_plan.md — each test encodes the TARGET state
of one plan item, written before the work is done.

Unlike tests/test_upgrade_plan_guards.py (regression pins that must stay green
the whole time), these tests are expected to be PENDING until their stage
lands:

  PEND  — the future module/API does not exist yet (stage not started).
  PASS  — the stage landed and matches the planned contract.
  FAIL  — the API exists but violates the contract (a real problem).

The exit code counts only FAILs, so this suite can run alongside the guards
from day one without polluting the signal. A stage item is "done" when its
test here flips from PEND to PASS (note it in the plan's Done note).

Where the plan leaves a name/signature to the implementer (e.g. the exact
function names in utils/report_finalize.py, the mask() signature), the test
asserts the names suggested in the plan — if you choose differently during
implementation, update the test in the same commit.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_upgrade_plan_acceptance.py
"""

import asyncio
import importlib
import inspect
import os
import sys

sys.path.insert(0, ".")


class Pending(Exception):
    """Raised when the target API of a plan item does not exist yet."""


def _module_or_pending(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise Pending(f"module '{name}' not present yet ({exc})")


def _attr_or_pending(module, attr: str):
    if not hasattr(module, attr):
        raise Pending(f"'{module.__name__}.{attr}' not present yet")
    return getattr(module, attr)


# ---------------------------------------------------------------------------
# Stage 1.1 — legacy free-form QMIX path deleted (D3)
# ---------------------------------------------------------------------------

def test_stage1_1_legacy_deleted():
    if os.path.exists("qmix_report_writer/graph/graph.py"):
        raise Pending("legacy graph/graph.py still present")

    # The graph package keeps working and no longer exposes QMIXGraph.
    import qmix_report_writer.graph as graph_pkg
    assert hasattr(graph_pkg, "Node")
    assert not hasattr(graph_pkg, "QMIXGraph")

    for legacy in ("experiments/run_qmix_eval.py", "scripts/query_qmix.py"):
        assert not os.path.exists(legacy), f"{legacy} should be deleted"

    # run_qmix_train.py is deleted in 1.1 and RECREATED in 4.6 — if present,
    # it must be the new one (no legacy QMIXGraph import).
    if os.path.exists("experiments/run_qmix_train.py"):
        with open("experiments/run_qmix_train.py", encoding="utf-8") as f:
            assert "QMIXGraph" not in f.read(), "old run_qmix_train still in place"

    import experiments.eval  # must keep importing standalone
    assert callable(experiments.eval.report_score)


# ---------------------------------------------------------------------------
# Stage 1.2 — config hygiene
# ---------------------------------------------------------------------------

def test_stage1_2_config_hygiene():
    from qmix_report_writer.utils.config import _load_defaults
    cfg = _load_defaults()
    if "evaluation" in cfg:
        raise Pending("stale 'evaluation:' section still in default.yaml")
    assert "num_rounds" not in (cfg.get("training") or {}), \
        "stale training.num_rounds survived the hygiene pass"


# ---------------------------------------------------------------------------
# Stage 2.2 — report finalization extracted + finalize flag on arun
# ---------------------------------------------------------------------------

def test_stage2_2_finalize_module_and_flag():
    module = _module_or_pending("qmix_report_writer.utils.report_finalize")
    # Suggested public names (plan 2.2) — rename here if you chose others.
    for fn in ("apply_citation_tags", "build_bibliography", "generate_abstract"):
        assert callable(_attr_or_pending(module, fn))

    from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
    params = inspect.signature(HandcraftedGraph.arun).parameters
    assert "finalize" in params, "arun() must gain the finalize flag (D6)"
    assert params["finalize"].default is True, "finalize must default True (handcrafted unchanged)"


# ---------------------------------------------------------------------------
# Stage 3.1 / 3.2 — RoundController seam (D2)
# ---------------------------------------------------------------------------

def test_stage3_controller_seam():
    module = _module_or_pending("qmix_report_writer.handcrafted_graph.controller")
    round_controller = _attr_or_pending(module, "RoundController")
    _attr_or_pending(module, "HandcraftedRoundController")

    for hook in ("round_plan", "on_round_end", "on_phase_start", "on_run_end"):
        assert hasattr(round_controller, hook), f"RoundController missing {hook}"

    from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
    params = inspect.signature(HandcraftedGraph.__init__).parameters
    assert "controller" in params, "HandcraftedGraph must accept controller="
    assert params["controller"].default is None, "controller must default to None (handcrafted default)"


# ---------------------------------------------------------------------------
# Stage 4.1 — action space v2 (8 actions, terminate dropped per D5)
# ---------------------------------------------------------------------------

def test_stage4_1_action_space_v2():
    from qmix_report_writer.qmix.agent_network import ACTION_NAMES, NUM_ACTIONS
    if NUM_ACTIONS == 9 and "terminate" in ACTION_NAMES:
        raise Pending("action space still v1 (9 actions incl. terminate)")

    assert NUM_ACTIONS == 8, f"expected 8 actions, got {NUM_ACTIONS}"
    assert "terminate" not in ACTION_NAMES
    assert ACTION_NAMES[0] == "no_op", "index 0 must be the true no-op (report 2.3)"
    assert ACTION_NAMES[7] == "append"


def test_stage4_1_action_masks():
    module = _module_or_pending("qmix_report_writer.qmix.action_masks")
    mask_fn = _attr_or_pending(module, "mask")
    from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
    from qmix_report_writer.handcrafted_graph.phases import PhaseType

    try:
        example = mask_fn(PhaseType.PLANNING, "Researcher", None)
    except TypeError as exc:
        raise Pending(f"mask() signature differs from the plan sketch — update this test ({exc})")

    assert len(example) == NUM_ACTIONS
    # append masked in every trained phase (PLANNING/RESEARCH per report 2.4;
    # DRAFTING per OD-1 choice (a): the write round stays scripted).
    for phase in (PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING):
        assert not mask_fn(phase, "Researcher", None)[7], f"append not masked in {phase}"
    # append never allowed for the Reviewer (its critique must not become prose).
    assert not mask_fn(PhaseType.RESEARCH, "Reviewer", None)[7]


# ---------------------------------------------------------------------------
# Stage 4.2 — QMIXRoundController exists (behavior via full-run 4.7/4.8)
# ---------------------------------------------------------------------------

def test_stage4_2_qmix_controller_module():
    module = _module_or_pending("qmix_report_writer.qmix.qmix_controller")
    controller_cls = _attr_or_pending(module, "QMIXRoundController")
    for hook in ("round_plan", "on_round_end"):
        assert hasattr(controller_cls, hook)


# ---------------------------------------------------------------------------
# Stage 4.3 — masked action selection in the trainer
# ---------------------------------------------------------------------------

def test_stage4_3_masked_select_actions():
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
    if "mask" not in inspect.signature(QMIXTrainer.select_actions).parameters:
        raise Pending("select_actions has no mask parameter yet")

    import torch
    n_agents, obs_dim = 5, 51
    trainer = QMIXTrainer(
        n_agents=n_agents, obs_dim=obs_dim, state_dim=n_agents * obs_dim + 3,
        gnn_hidden_dim=32, gnn_layers=1, rnn_hidden_dim=32, mixing_hidden_dim=16,
    )
    obs = torch.randn(n_agents, obs_dim)
    adj = torch.eye(n_agents)
    hidden = trainer.agent_network.init_hidden(n_agents)

    only_action_2 = torch.zeros(trainer.n_acting_agents, trainer.n_actions, dtype=torch.bool)
    only_action_2[:, 2] = True
    for epsilon in (0.0, 1.0):  # both greedy AND the random draw must respect the mask
        actions, _ = trainer.select_actions(obs, adj, hidden, epsilon=epsilon, mask=only_action_2)
        assert bool((actions == 2).all()), f"mask violated at epsilon={epsilon}: {actions.tolist()}"


# ---------------------------------------------------------------------------
# Stage 4.5 — HandcraftedPromptSet renders the QMIX action line
# ---------------------------------------------------------------------------

def test_stage4_5_prompt_action_render():
    from qmix_report_writer.handcrafted_graph.state import PhaseState
    from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry
    PhaseState._instance = None  # fresh -> PLANNING

    prompt_set = PromptSetRegistry.get("handcrafted_redacting")
    without_action = prompt_set.get_context_block("Lead Architect")
    with_action = prompt_set.get_context_block("Lead Architect", action=1)  # broadcast_all
    if with_action == without_action:
        raise Pending("get_context_block does not render the action yet")

    assert "broadcast" in with_action, "action description missing from the block"
    assert "Round" in with_action, "phase context must still render alongside the action"
    assert "broadcast" not in without_action, "no action given -> no action line"


# ---------------------------------------------------------------------------
# Stage 4.6 — new runners
# ---------------------------------------------------------------------------

def test_stage4_6_runner_module():
    module = _module_or_pending("qmix_report_writer.qmix.runner")
    assert callable(_attr_or_pending(module, "run_qmix_train"))
    assert callable(_attr_or_pending(module, "run_qmix"))


# ---------------------------------------------------------------------------
# Stage 6.2 — TechnicalWriter deleted (OD-2 choice (a))
# ---------------------------------------------------------------------------

def test_stage6_2_technical_writer_deleted():
    if os.path.exists("qmix_report_writer/agents/technical_writer.py"):
        raise Pending("agents/technical_writer.py still present")

    import qmix_report_writer.agents as agents_pkg  # __init__ must import cleanly
    from qmix_report_writer.agents.agent_registry import AgentRegistry
    assert "TechnicalWriter" not in list(AgentRegistry.keys())
    assert not hasattr(agents_pkg, "TechnicalWriter")


# ---------------------------------------------------------------------------
# Stage 6.4 — benchmark dataset loaders pruned (OD-3 choice (a))
# ---------------------------------------------------------------------------

def test_stage6_4_datasets_pruned():
    if os.path.exists("datasets/mmlu_dataset.py"):
        raise Pending("benchmark dataset loaders still present")

    for loader in ("gaia_dataset.py", "hle_dataset.py", "humaneval_dataset.py",
                   "livecodebench_dataset.py", "math_dataset.py"):
        assert not os.path.exists(f"datasets/{loader}"), f"datasets/{loader} should be deleted"

    from datasets.tasks import tasks  # the training task list must survive
    assert isinstance(tasks, list) and len(tasks) > 0
    import datasets  # package __init__ must import cleanly post-pruning
    assert datasets is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all():
    passed = pending = failed = 0
    cases = [
        ("test_stage1_1_legacy_deleted", test_stage1_1_legacy_deleted),
        ("test_stage1_2_config_hygiene", test_stage1_2_config_hygiene),
        ("test_stage2_2_finalize_module_and_flag", test_stage2_2_finalize_module_and_flag),
        ("test_stage3_controller_seam", test_stage3_controller_seam),
        ("test_stage4_1_action_space_v2", test_stage4_1_action_space_v2),
        ("test_stage4_1_action_masks", test_stage4_1_action_masks),
        ("test_stage4_2_qmix_controller_module", test_stage4_2_qmix_controller_module),
        ("test_stage4_3_masked_select_actions", test_stage4_3_masked_select_actions),
        ("test_stage4_5_prompt_action_render", test_stage4_5_prompt_action_render),
        ("test_stage4_6_runner_module", test_stage4_6_runner_module),
        ("test_stage6_2_technical_writer_deleted", test_stage6_2_technical_writer_deleted),
        ("test_stage6_4_datasets_pruned", test_stage6_4_datasets_pruned),
    ]
    for name, fn in cases:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            print(f"PASS  {name}")
            passed += 1
        except Pending as why:
            print(f"PEND  {name} — {why}")
            pending += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {pending} pending, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
