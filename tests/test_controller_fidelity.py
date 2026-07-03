"""
Translation-fidelity test (upgrade-plan Stage 4.7).

Proves the QMIX action→plan translation and the controller seam are faithful:
a controller that expresses the handcrafted phase tables THROUGH the QMIX
action vocabulary must produce, round for round, the same active-agent sets
and the same message edges as the default HandcraftedRoundController.

Both runs are fully offline (stub RAG, scripted mock LLMs — same fixtures as
test_training_mode_shape) over the training phase list, so the only variable
is the controller.

Vocabulary encoding of the tables (acting order LA, R, DA, Rev; selective
target j = action 2+j):
    PLANNING r0/r1 : R=selective(LA)=2,  LA=aggregate=6,  DA=Rev=no_op
    RESEARCH A     : LA=selective(R)=3,  R=aggregate=6,   DA=Rev=no_op
    RESEARCH B     : R=broadcast=1, LA=selective(DA)=4, DA=aggregate=6, Rev=no_op
    DRAFTING prep  : LA=selective(DA)=4, R=selective(DA)=4, DA=aggregate=6, Rev=no_op
    DRAFTING write : scripted in BOTH runs (blueprint reuse) — controller not consulted.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_controller_fidelity.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from unittest.mock import patch

from qmix_report_writer.utils.globals import (
    ReportState, SourceBuffer, ExecutionTrace, PromptTokens, CompletionTokens,
)
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import (
    PhaseType, PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE,
)
from qmix_report_writer.handcrafted_graph.controller import RoundController
from qmix_report_writer.qmix.qmix_controller import plan_from_actions

AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
ACTING = AGENTS[:-1]
TASK = "Controlled fission and fusion reactions"

_RESPONSES = {
    None: "fission energy release neutron\nreactor control rod materials\nfusion plasma confinement",
    "Researcher": "[Evidence] Fission releases about 200 MeV per event | [stub.pdf]",
    "LeadArchitect": (
        "Outline grounded in confirmed coverage:\n"
        "1. **Fission Fundamentals**\n"
        "2. **Fusion Confinement**\n"
        "<task>Gather evidence for the current section.</task>"
    ),
    "DataAnalyst": "- Claim: fission releases ~200 MeV per event (stub.pdf, p.1)",
    "Collector": "## Stub Section\n\nFission releases approximately 200 MeV per event.",
    "Reviewer": "[NO_REVISION_NEEDED]",
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


# Handcrafted tables expressed in the QMIX action vocabulary,
# keyed by (phase, pattern index = round_idx % n_patterns).
_ACTION_TABLE = {
    (PhaseType.PLANNING, 0): {"LeadArchitect": 6, "Researcher": 2, "DataAnalyst": 0, "Reviewer": 0},
    (PhaseType.PLANNING, 1): {"LeadArchitect": 6, "Researcher": 2, "DataAnalyst": 0, "Reviewer": 0},
    (PhaseType.RESEARCH, 0): {"LeadArchitect": 3, "Researcher": 6, "DataAnalyst": 0, "Reviewer": 0},
    (PhaseType.RESEARCH, 1): {"LeadArchitect": 4, "Researcher": 1, "DataAnalyst": 6, "Reviewer": 0},
    (PhaseType.DRAFTING, 0): {"LeadArchitect": 4, "Researcher": 4, "DataAnalyst": 6, "Reviewer": 0},
    (PhaseType.DRAFTING, 1): {"LeadArchitect": 0, "Researcher": 2, "DataAnalyst": 6, "Reviewer": 0},
}


class _VocabController(RoundController):
    """Emits the handcrafted plans through plan_from_actions()."""

    async def round_plan(self, phase, round_idx, topology, nodes, task_input):
        pattern = round_idx % 2
        actions = _ACTION_TABLE[(phase, pattern)]
        return plan_from_actions(actions, ACTING, topology)


def _reset_singletons():
    for cls in (ReportState, SourceBuffer, ExecutionTrace, PhaseState,
                PromptTokens, CompletionTokens):
        cls._instance = None


async def _run(controller) -> list:
    """Run the training-shape pipeline; return per-round (active, edges) sets."""
    _reset_singletons()
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

    await graph.arun({"task": TASK}, max_validation_attempts=0, finalize=False)

    rounds = []
    for round_data in ExecutionTrace.instance().trace:
        active = frozenset(
            a for a in round_data.get("exec_order", []) if a in AGENTS
        )
        edges = frozenset(
            (sender, receiver)
            for sender in AGENTS if sender in round_data
            for receiver in round_data[sender].get("message_to", [])
            if receiver in AGENTS
        )
        rounds.append((active, edges))
    return rounds


async def test_vocab_controller_matches_handcrafted():
    baseline = await _run(None)                  # default HandcraftedRoundController
    vocab = await _run(_VocabController())

    assert len(baseline) == len(vocab), \
        f"round counts differ: {len(baseline)} vs {len(vocab)}"
    for i, ((b_active, b_edges), (v_active, v_edges)) in enumerate(zip(baseline, vocab)):
        assert b_active == v_active, \
            f"round {i}: active sets differ\n  handcrafted: {sorted(b_active)}\n  vocab:       {sorted(v_active)}"
        assert b_edges == v_edges, \
            f"round {i}: edges differ\n  handcrafted: {sorted(b_edges)}\n  vocab:       {sorted(v_edges)}"

    # Same final report shape on both paths.
    assert len(ReportState.instance().sections) == 2
    print(f"PASS  test_vocab_controller_matches_handcrafted ({len(baseline)} rounds identical)")


async def _run_all():
    passed = failed = 0
    for name, fn in [("test_vocab_controller_matches_handcrafted",
                      test_vocab_controller_matches_handcrafted)]:
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
