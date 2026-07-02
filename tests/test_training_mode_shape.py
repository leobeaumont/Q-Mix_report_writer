"""
Training-mode run-shape verification (upgrade-plan Stage 3.3).

Drives a full HandcraftedGraph.arun() offline — stub RAG, scripted mock LLMs,
no Ollama, no ChromaDB — in the exact configuration the QMIX training runner
will use (decision D6):

    phases=[PLANNING, RESEARCH, DRAFTING]
    max_validation_attempts=0
    finalize=False

and asserts the run shape:
  * sections get planned and written,
  * no SECTION_REVIEW / VALIDATION round ever executes (no Reviewer call),
  * no bibliography and no abstract are produced,
  * the validation-loop bookkeeping survives an empty correction-phase list.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_training_mode_shape.py
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

AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
TASK = "Controlled fission and fusion reactions"

# Scripted responses per calling_agent (None = the Researcher's RAG-query
# formulation call, which passes no calling_agent).
_RESPONSES = {
    None: (
        "fission energy release neutron\n"
        "reactor control rod materials\n"
        "fusion plasma confinement"
    ),
    "Researcher": (
        "[Evidence] Fission releases about 200 MeV per event | [stub.pdf]\n"
        "[Evidence] Control rods absorb thermal neutrons | [stub.pdf]"
    ),
    "LeadArchitect": (
        "Outline grounded in confirmed coverage:\n"
        "1. **Fission Fundamentals**\n"
        "2. **Fusion Confinement**\n"
        "<task>Gather evidence for the current section.</task>"
    ),
    "DataAnalyst": (
        "- Claim: fission releases ~200 MeV per event (stub.pdf, p.1)\n"
        "- Claim: control rods absorb thermal neutrons (stub.pdf, p.2)"
    ),
    "Collector": (
        "## Stub Section\n\n"
        "Fission releases approximately 200 MeV per fission event, and control "
        "rods regulate the chain reaction by absorbing thermal neutrons."
    ),
    "Reviewer": "[NO_REVISION_NEEDED]",  # must never be requested in this test
}


class _ScriptedLLM:
    """Dispatches canned responses by calling_agent and counts calls."""

    def __init__(self, log: list):
        self._log = log

    async def agen(self, messages, calling_agent=None, **kwargs):
        self._log.append(calling_agent)
        return _RESPONSES[calling_agent]

    def gen(self, messages, calling_agent=None, **kwargs):
        self._log.append(calling_agent)
        return _RESPONSES[calling_agent]


class _StubRAG:
    """Stands in for RAGManager: unique chunk ids per call, fixed content."""

    def __init__(self, *args, **kwargs):
        self._counter = 0

    def query_docs_multi(self, queries, top_k=5):
        self._counter += 1
        return [
            {
                "id": f"chunk_{self._counter}_{i}",
                "source": "stub.pdf",
                "page": i + 1,
                "content": (
                    "Fission releases about 200 MeV per event; control rods "
                    "absorb thermal neutrons to regulate the chain reaction."
                ),
            }
            for i in range(2)
        ]


def _reset_singletons():
    for cls in (ReportState, SourceBuffer, ExecutionTrace, PhaseState,
                PromptTokens, CompletionTokens):
        cls._instance = None


async def test_training_mode_run_shape():
    _reset_singletons()

    with patch("qmix_report_writer.agents.researcher.RAGManager", _StubRAG):
        from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
        graph = HandcraftedGraph(
            llm_name="tinyllama",
            agent_names=AGENTS,
            execution_trace=True,
            phases=[PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE],
        )

    call_log: list = []
    for node in graph.nodes.values():
        node.llm = _ScriptedLLM(call_log)

    answers, _ = await graph.arun(
        {"task": TASK},
        max_validation_attempts=0,
        finalize=False,
    )
    report = answers[0]
    rs = ReportState.instance()

    # Outline was parsed and drove code-driven section iteration.
    assert rs.planned_sections == ["Fission Fundamentals", "Fusion Confinement"], \
        f"planned_sections: {rs.planned_sections}"
    assert len(rs.sections) == 2, f"expected 2 written sections, got {len(rs.sections)}"
    assert "## Stub Section" in report

    # Correction stages never ran: phase history stops at DRAFTING...
    state = PhaseState.instance()
    assert state.current_phase == PhaseType.DRAFTING
    assert PhaseType.SECTION_REVIEW not in state.phase_history
    assert PhaseType.VALIDATION not in state.phase_history
    # ...the Reviewer was never called...
    assert "Reviewer" not in call_log, "Reviewer executed in training mode"
    for round_data in ExecutionTrace.instance().trace:
        assert "Reviewer" not in round_data.get("exec_order", [])

    # ...and finalization was skipped (D6).
    assert rs.bibliography == "", "bibliography built despite finalize=False"
    assert "## Abstract" not in report
    assert "## Bibliography" not in report

    print("PASS  test_training_mode_run_shape")


async def _run_all():
    passed = failed = 0
    for name, fn in [("test_training_mode_run_shape", test_training_mode_run_shape)]:
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
