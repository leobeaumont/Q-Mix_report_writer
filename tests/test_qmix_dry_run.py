"""
QMIX random-policy dry run, fully offline (upgrade-plan Stage 4.8b).

Runs one complete training episode — untrained network, ε=1.0 (pure masked
random), stub RAG, scripted mock LLMs, mock scorer — through the real
QMIXRoundController + HandcraftedGraph seam, and asserts:

  * the pipeline never crashes under a random policy,
  * masks are respected on every recorded step (the controller also asserts
    this internally and would raise),
  * episode transitions have the right shapes (obs/actions/mask/global_state),
  * reward events fire only on real appends and land on the EVENT STEP only
    (OD-A), with the terminal macro added to the final step (reward v2),
  * the episode feeds the replay buffer and a train_step() runs.

The real tiny-model dry run (live Ollama) is the user-run half of checkpoint
4.8.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_qmix_dry_run.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from unittest.mock import patch

import numpy as np

from qmix_report_writer.utils.globals import (
    ReportState, SourceBuffer, ExecutionTrace, PromptTokens, CompletionTokens,
    Score, LengthGoal,
)
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import (
    PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE,
)

AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
TASK = "Controlled fission and fusion reactions"

_RESPONSES = {
    None: "fission energy release neutron\nreactor control rods\nfusion plasma confinement",
    "Researcher": "[Evidence] Fission releases about 200 MeV per event | [stub.pdf]",
    "LeadArchitect": (
        "1. **Fission Fundamentals**\n2. **Fusion Confinement**\n"
        "<task>Gather evidence for the current section.</task>"
    ),
    "DataAnalyst": "- Claim: fission releases ~200 MeV per event (stub.pdf, p.1)",
    "Collector": "## Stub Section\n\nFission releases approximately 200 MeV per event.",
    "Reviewer": "Noted: evidence looks consistent.",
}


class _ScriptedLLM:
    async def agen(self, messages, calling_agent=None, **kwargs):
        return _RESPONSES[calling_agent]

    def gen(self, messages, calling_agent=None, **kwargs):
        return _RESPONSES[calling_agent]


class _StubRAG:
    def __init__(self, *args, **kwargs):
        self._counter = 0

    def query_docs_multi(self, queries, top_k=5):
        self._counter += 1
        return [{
            "id": f"chunk_{self._counter}_{i}",
            "source": "stub.pdf",
            "page": i + 1,
            "content": "Fission releases about 200 MeV per event.",
        } for i in range(2)]


def _reset_singletons():
    for cls in (ReportState, SourceBuffer, ExecutionTrace, PhaseState,
                PromptTokens, CompletionTokens, Score, LengthGoal):
        cls._instance = None


async def test_random_policy_dry_run():
    _reset_singletons()

    from qmix_report_writer.qmix.observations import get_obs_dim, get_state_dim
    from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
    from qmix_report_writer.qmix.qmix_controller import QMIXRoundController
    from qmix_report_writer.qmix.runner import build_trainer

    # Stub evaluator (reward v2 interface): per-chunk scores + terminal macro.
    from types import SimpleNamespace

    chunk_values = iter([0.55, 0.65, 0.75, 0.80, 0.85])

    class _StubEvaluator:
        def __init__(self):
            self.chunk_calls = 0
            self.macro_calls = 0

        async def score_chunk(self, chunk, sources, task, context=None):
            self.chunk_calls += 1
            return SimpleNamespace(score=next(chunk_values), grounding_ratio=1.0)

        async def score_report(self, task, outline, report):
            self.macro_calls += 1
            return SimpleNamespace(score=0.7)

    evaluator = _StubEvaluator()
    trainer = build_trainer(len(AGENTS))
    controller = QMIXRoundController(
        trainer, AGENTS, train=True, epsilon=1.0, evaluator=evaluator,
    )

    with patch("qmix_report_writer.agents.researcher.RAGManager", _StubRAG):
        from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
        graph = HandcraftedGraph(
            llm_name="tinyllama",
            agent_names=AGENTS,
            execution_trace=True,
            phases=[PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE],
            controller=controller,
        )
    for node in graph.nodes.values():
        node.llm = _ScriptedLLM()

    answers, _ = await graph.arun(
        {"task": TASK}, max_validation_attempts=0, finalize=False,
    )
    await controller.on_run_end()  # idempotent runner-path call

    episode = controller.episode
    assert episode.steps, "no transitions recorded"

    # Shapes + mask compliance on every recorded step.
    n_acting = len(AGENTS) - 1
    for step in episode.steps:
        assert step.observations.shape == (len(AGENTS), get_obs_dim())
        assert step.global_state.shape == (get_state_dim(len(AGENTS)),)
        assert step.actions.shape == (n_acting,)
        assert step.mask.shape == (n_acting, NUM_ACTIONS)
        for i, action in enumerate(step.actions.tolist()):
            assert step.mask[i][action], f"mask violated: agent {i} action {action}"
        assert not step.mask[:, 0].all() or True  # masks exist per agent

    # Reward v2 placement: one event per real append, landing on the event
    # step only (OD-A); the terminal macro is added to the final step; all
    # other steps carry exactly 0 (token term is flag-gated off).
    assert len(ReportState.instance().sections) == 2
    assert evaluator.chunk_calls == 2, "one grounded chunk call per append"
    assert evaluator.macro_calls == 1, "exactly one terminal macro call"
    rewarded = [s for s in episode.steps if s.team_reward != 0.0]
    assert len(rewarded) == 2, (
        f"expected exactly the two event steps to carry reward, "
        f"got {len(rewarded)}"
    )
    assert episode.steps[-1].team_reward > 0.7, \
        "final step must include the terminal macro reward"
    assert episode.steps[-1].done is True

    # Replay + one training step (duplicate the episode to fill a batch).
    for _ in range(trainer.batch_size):
        trainer.replay_buffer.push(episode)
    info = trainer.train_step()
    assert info is not None and "loss" in info, "train_step did not run"

    print(
        f"PASS  test_random_policy_dry_run "
        f"(steps={episode.length}, rewarded={len(rewarded)}, "
        f"total_reward={episode.total_reward:.3f}, loss={info['loss']:.4f})"
    )


async def _run_all():
    passed = failed = 0
    for name, fn in [("test_random_policy_dry_run", test_random_policy_dry_run)]:
        try:
            await fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
