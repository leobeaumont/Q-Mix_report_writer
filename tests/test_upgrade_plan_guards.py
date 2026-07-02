"""
Guard tests for qmix_upgrade_plan.md — written BEFORE the changes, as side
validation while the plan is executed. Each section names the plan item it
guards. Two kinds of tests:

  * Regression pins: current behavior that the refactor stages (1-3) must
    preserve. If a pin breaks during a stage, behavior drifted.
  * Contract pins: current semantics that new Stage-4 code builds on (score
    deltas, append/skip sentinels, action-space structure).

Deliberately simple — anything needing a real LLM, ChromaDB, or a full arun()
is left to the plan's full-run checkpoints (0.1, 2.5, 3.3/3.4, 4.7, 4.8).

Notes for the executor of the plan:
  * Stage 2.2/2.3/2.4 move functions to new modules: update the IMPORTS here
    (marked with "Stage 2.x:"), keep the assertions.
  * Stage 4.1 changes the action list: test_action_space_consistency and
    test_trainer_select_actions_shape assert structure, not the action count,
    so they must stay green unchanged.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_upgrade_plan_guards.py
"""

import asyncio
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from unittest.mock import AsyncMock, MagicMock

from qmix_report_writer.utils.globals import (
    ReportState, SourceBuffer, ExecutionTrace, Score, LengthGoal,
)
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import (
    PhaseType, PHASE_SEQUENCE, PHASE_MAP,
)
# Stage 2.2 / 2.4: these move out of graph.py — update imports only.
from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
from qmix_report_writer.handcrafted_graph.prompts.handcrafted_prompt_set import (
    HandcraftedPromptSet, _extract_section_directive,
)
from qmix_report_writer.handcrafted_graph.scheduler import RoundScheduler, SkipStrategy
from qmix_report_writer.graph.node import Node
from qmix_report_writer.agents.collector import Collector, _ABSENCE_RE, _SENTINEL_OUTPUTS
from qmix_report_writer.agents.data_analyst import DataAnalyst
from qmix_report_writer.agents.lead_architect import LeadArchitect
from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry
from qmix_report_writer.utils.report_filter import filter_meta_commentary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_singletons():
    for cls in (ReportState, SourceBuffer, ExecutionTrace, Score, LengthGoal, PhaseState):
        cls._instance = None


def _bare_graph() -> HandcraftedGraph:
    """A HandcraftedGraph without agents/LLMs — enough for its pure methods."""
    graph = HandcraftedGraph.__new__(HandcraftedGraph)
    graph.id = "test"
    return graph


def _mock_llm(responses: list[str]) -> MagicMock:
    seq_sync = iter(responses)
    seq_async = iter(responses)
    llm = MagicMock()
    llm.gen = MagicMock(side_effect=lambda *a, **kw: next(seq_sync))
    llm.agen = AsyncMock(side_effect=lambda *a, **kw: next(seq_async))
    return llm


def _bare_agent(cls, role: str, llm: MagicMock):
    agent = cls.__new__(cls)
    agent.role = role
    agent.llm = llm
    agent.report = ReportState.instance()
    agent.prompt_set = PromptSetRegistry.get("redacting")
    if cls is Collector:
        agent.source_buffer = SourceBuffer.instance()
    return agent


class _DummyNode(Node):
    """Minimal concrete Node for scheduler tests."""

    def _execute(self, *a, **kw):
        pass

    async def _async_execute(self, *a, **kw):
        pass

    def _process_inputs(self, *a, **kw):
        pass


# ---------------------------------------------------------------------------
# Stage 1.1 — legacy deletion guards
# ---------------------------------------------------------------------------

def test_node_import_and_eval_standalone():
    # graph package must keep exporting Node after graph.py (QMIXGraph) is deleted.
    from qmix_report_writer.graph import Node as _Node
    assert _Node is Node

    # experiments/eval.py must import standalone (it survives Stage 1.1 and is
    # reused by the Stage 4.4 reward hook).
    from experiments.eval import length_score, report_score
    assert callable(report_score)

    _reset_singletons()
    rs = ReportState.instance()
    rs.content = "x" * 25000
    assert abs(length_score(25000, 8500) - 1.0) < 1e-9
    rs.content = ""
    assert length_score(25000, 8500) < 0.05
    print("PASS  test_node_import_and_eval_standalone")


# ---------------------------------------------------------------------------
# Stage 2.1 — sync/async consolidation guards (behavior equivalence)
# ---------------------------------------------------------------------------

def _captured_messages(mock_fn):
    (messages,), _ = mock_fn.call_args
    return messages


async def test_data_analyst_sync_async_equivalence():
    spatial = {"r_1": {"role": "Researcher", "output": "Evidence: X is 5 GeV."}}
    temporal = {"d_2": {"role": "Data Analyst", "output": "Prior notes."}}
    task = {"task": "Write a report on X"}

    _reset_singletons()
    llm = _mock_llm(["SYNC_RESP"])
    da = _bare_agent(DataAnalyst, "Data Analyst", llm)
    sync_out = da._execute(task, spatial, temporal, action=3)
    sync_msgs = _captured_messages(llm.gen)

    _reset_singletons()
    llm = _mock_llm(["SYNC_RESP"])
    da = _bare_agent(DataAnalyst, "Data Analyst", llm)
    async_out = await da._async_execute(task, spatial, temporal, action=3)
    async_msgs = _captured_messages(llm.agen)

    assert sync_out == async_out == "SYNC_RESP"
    assert sync_msgs == async_msgs, "sync and async paths built different prompts"
    # The QMIX action context block must render identically on both paths.
    assert "Current Action" in async_msgs[1]["content"]
    print("PASS  test_data_analyst_sync_async_equivalence")


async def test_lead_architect_sync_async_equivalence_and_parse():
    response = "Plan is ready.\n<task>Prepare the Introduction evidence.</task>"
    task = {"task": "Write a report on X"}
    spatial = {"r_1": {"role": "Researcher", "output": "Coverage: topics A, B."}}

    _reset_singletons()
    la = _bare_agent(LeadArchitect, "Lead Architect", _mock_llm([response]))
    sync_out = la._execute(task, spatial, {})
    sync_task = ReportState.instance().task

    _reset_singletons()
    la = _bare_agent(LeadArchitect, "Lead Architect", _mock_llm([response]))
    async_out = await la._async_execute(task, spatial, {})
    async_task = ReportState.instance().task

    assert sync_out == async_out
    assert sync_task == async_task == "Prepare the Introduction evidence."

    # _parse_response contract pins.
    assert la._parse_response("done [DRAFTING_COMPLETE]")[0] == "[DRAFTING_COMPLETE]"
    assert la._parse_response("ok [REVISION_COMPLETE]")[0] == "[REVISION_COMPLETE]"
    fallback_task, _ = la._parse_response("no tag here")
    assert fallback_task.startswith("Continue developing")
    print("PASS  test_lead_architect_sync_async_equivalence_and_parse")


# ---------------------------------------------------------------------------
# Stage 2.2 — report finalization pins (citations / bibliography)
# ---------------------------------------------------------------------------

def test_citation_overlap_tagging():
    _reset_singletons()
    rs = ReportState.instance()
    chunk = {
        "id": "c1",
        "source": "2105.06979.pdf",
        "page": 7,
        "content": (
            "The neutron star maximum mass depends on the equation of state "
            "stiffness parameters derived from dense matter."
        ),
    }
    body = (
        "## Equation of State\n\n"
        "The neutron star maximum mass depends strongly on the equation of "
        "state stiffness parameters."
    )
    rs.append(body, "progress", [chunk])

    _bare_graph()._apply_citation_tags(0)

    content = rs.sections[0]["content"]
    assert "[cite:1, p.7]" in content, f"tag missing: {content!r}"
    assert content.splitlines()[0] == "## Equation of State"  # heading untouched
    assert rs.bibliography_map == {"2105.06979.pdf": 1}
    assert rs.citation_counts.get(1, 0) >= 1
    print("PASS  test_citation_overlap_tagging")


def test_inline_reference_rewrite_and_orphans():
    _reset_singletons()
    rs = ReportState.instance()
    graph = _bare_graph()
    known = {"2105.06979.pdf": "2105.06979.pdf"}

    out, n = graph._rewrite_inline_references(
        "As shown in [2105.06979.pdf | Page: 26] the flow is anisotropic.",
        known, rs.bibliography_map, rs,
    )
    assert n == 1 and "[cite:1, p.26]" in out, f"Got: {out!r}"

    out, n = graph._rewrite_inline_references(
        "As shown in [unknown_file.pdf | Page: 3] nothing changes.",
        known, rs.bibliography_map, rs,
    )
    assert n == 0 and "[unknown_file.pdf | Page: 3]" in out

    out, n = graph._strip_orphan_citation_markers("This agrees with Ref. [32] closely.")
    assert n == 1 and "[32]" not in out
    out, n = graph._strip_orphan_citation_markers("Verified value [cite:2, p.4].")
    assert n == 0 and "[cite:2, p.4]" in out
    print("PASS  test_inline_reference_rewrite_and_orphans")


def test_bibliography_build():
    _reset_singletons()
    rs = ReportState.instance()
    rs.sources = [
        {"source": "2105.06979.pdf", "title": "Dense Matter", "author": "A. Author", "year": "2021"},
        {"source": "notes.pdf"},
    ]
    rs.bibliography_map = {"2105.06979.pdf": 1}
    rs.citation_counts = {1: 2}

    _bare_graph()._build_bibliography()

    bib = rs.bibliography
    assert bib.startswith("## Bibliography")
    assert "[1]" in bib and "arXiv:2105.06979" in bib and "(2 citations)" in bib
    assert "Dense Matter" in bib and "A. Author" in bib
    assert "### Consulted Sources" in bib and "notes.pdf" in bib
    print("PASS  test_bibliography_build")


# ---------------------------------------------------------------------------
# Stage 2.3 — trace round-slot schema pin
# ---------------------------------------------------------------------------

def test_trace_round_schema():
    _reset_singletons()
    agent_names = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]
    graph = _bare_graph()
    graph.agent_names = agent_names
    graph.collector_id = "Collector_4"
    graph.execution_trace = ExecutionTrace.instance()

    graph._init_trace_round()

    round_data = ExecutionTrace.instance().trace[-1]
    for name in agent_names:
        slot = round_data[name]
        for key in ("action", "message_to", "prompt", "response", "time", "completion_tokens"):
            assert key in slot, f"{name} slot missing {key}"
    assert "sources" in round_data["RAG"]
    assert "PBDS" in round_data
    assert round_data["exec_order"] == []
    assert "report_state" in round_data["Collector"]
    print("PASS  test_trace_round_schema")


# ---------------------------------------------------------------------------
# Stage 2.4 — validation-loop code-motion pins
# ---------------------------------------------------------------------------

def test_revalidation_sections_and_directive_extract():
    sections = [{"id": f"section_{i}", "title": f"S{i}", "content": "x"} for i in range(1, 5)]
    state = SimpleNamespace(
        validation_directive="- section_2: use the 5 GeV value everywhere.",
        validation_issues="section_3 contradicts the earlier statement.",
        sections=sections,
    )
    picked = HandcraftedGraph._revalidation_sections(state)
    assert [s["id"] for s in picked] == ["section_2", "section_3"]

    state_no_ids = SimpleNamespace(validation_directive="", validation_issues="all fine", sections=sections)
    assert HandcraftedGraph._revalidation_sections(state_no_ids) == sections

    directive = (
        "- section_1: replace 10 GeV with 5 GeV.\n"
        "- section_3: shorten the derivation,\n  keep only the result.\n"
    )
    assert _extract_section_directive(directive, "section_1") == "replace 10 GeV with 5 GeV."
    assert "shorten the derivation" in _extract_section_directive(directive, "section_3")
    assert _extract_section_directive(directive, "section_2") == ""
    print("PASS  test_revalidation_sections_and_directive_extract")


# ---------------------------------------------------------------------------
# Stage 3.2 — pins for what the default RoundController must reproduce
# ---------------------------------------------------------------------------

def test_phase_sequence_invariants():
    order = [p.name for p in PHASE_SEQUENCE]
    assert order == [
        PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING,
        PhaseType.SECTION_REVIEW, PhaseType.VALIDATION,
    ]
    for earlier, later in zip(PHASE_SEQUENCE, PHASE_SEQUENCE[1:]):
        assert earlier.next_phase == later.name
    assert PHASE_MAP[PhaseType.SECTION_REVIEW].section_aware
    assert PHASE_MAP[PhaseType.VALIDATION].window_aware

    planning_r0 = PHASE_MAP[PhaseType.PLANNING].round_topologies[0]
    assert ("Researcher", "LeadArchitect") in planning_r0.edges
    assert {"Researcher", "LeadArchitect"} <= set(planning_r0.required_agents)

    drafting = PHASE_MAP[PhaseType.DRAFTING].round_topologies
    assert len(drafting) == 2
    assert ("DataAnalyst", "Collector") in drafting[1].edges
    print("PASS  test_phase_sequence_invariants")


def test_scheduler_temporal_heuristic():
    productive = _DummyNode(id="a", agent_name="A")
    productive.last_memory["outputs"] = ["real findings"]
    holding = _DummyNode(id="b", agent_name="B")
    holding.last_memory["outputs"] = ["[HOLD]"]
    silent = _DummyNode(id="c", agent_name="C")

    nodes = {"a": productive, "b": holding, "c": silent}
    scheduler = RoundScheduler(nodes, None, SkipStrategy.TEMPORAL_HEURISTIC)

    assert scheduler._temporal_heuristic(productive) is True
    assert scheduler._temporal_heuristic(holding) is False
    assert scheduler._temporal_heuristic(silent) is False

    # Seed condition: when NO node has output (phase boundary), include everyone.
    productive.last_memory["outputs"] = []
    holding.last_memory["outputs"] = []
    assert scheduler._temporal_heuristic(silent) is True
    print("PASS  test_scheduler_temporal_heuristic")


def test_drafting_blueprint_usable():
    usable = HandcraftedGraph._drafting_blueprint_is_usable
    assert usable("- Claim: X rises with Y (source p.3)\n- State Deficiency: exact ratio") is True
    assert usable("[NO NEW EVIDENCE]") is False
    assert usable("State Deficiency: melting point\nState Deficiency: density") is False
    assert usable("[RESEARCH_EXHAUSTED] RAG returned no documents.") is False
    assert usable("") is False
    print("PASS  test_drafting_blueprint_usable")


def test_parse_section_titles():
    parse = HandcraftedGraph._parse_section_titles
    bold = "1. **Introduction**\n2. **Methods and Data**\n3. **Results**"
    assert parse(bold) == ["Introduction", "Methods and Data", "Results"]
    plain = "1) Introduction\n2) Methods\n3) Results."
    assert parse(plain) == ["Introduction", "Methods", "Results"]
    assert parse("No outline here, only prose.") == []
    print("PASS  test_parse_section_titles")


# ---------------------------------------------------------------------------
# Stage 4.1 / 4.3 — action space & trainer selection structural pins
# (assert structure, not the action count: must survive the 9 -> 8 change)
# ---------------------------------------------------------------------------

def test_action_space_consistency():
    from qmix_report_writer.qmix.agent_network import ACTION_NAMES, NUM_ACTIONS
    assert NUM_ACTIONS == len(ACTION_NAMES)
    assert len(set(ACTION_NAMES)) == len(ACTION_NAMES), "duplicate action names"
    print("PASS  test_action_space_consistency")


def test_trainer_select_actions_shape():
    import torch
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer

    n_agents, obs_dim = 5, 51
    state_dim = n_agents * obs_dim + 3
    trainer = QMIXTrainer(
        n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim,
        gnn_hidden_dim=32, gnn_layers=1, rnn_hidden_dim=32, mixing_hidden_dim=16,
    )
    obs = torch.randn(n_agents, obs_dim)
    adj = torch.eye(n_agents)
    hidden = trainer.agent_network.init_hidden(n_agents)

    for epsilon in (0.0, 1.0):
        actions, new_hidden = trainer.select_actions(obs, adj, hidden, epsilon=epsilon)
        assert actions.shape == (trainer.n_acting_agents,)
        assert actions.dtype == torch.long
        assert bool(((actions >= 0) & (actions < trainer.n_actions)).all())
        assert new_hidden.shape == hidden.shape
    print("PASS  test_trainer_select_actions_shape")


# ---------------------------------------------------------------------------
# Stage 4.4 — reward-hook contract pins (score deltas + append/skip mechanics)
# ---------------------------------------------------------------------------

def test_score_delta_semantics():
    _reset_singletons()
    score = Score.instance()
    score.update(0.5)
    assert score.get_delta() == 0.5          # first update: delta = current value
    score.update(0.7)
    assert abs(score.get_delta() - 0.2) < 1e-9

    goal = LengthGoal.instance()
    goal.update(0.3)
    goal.update(0.25)
    assert abs(goal.get_delta() - (-0.05)) < 1e-9
    print("PASS  test_score_delta_semantics")


def test_collector_append_and_skip():
    task = {"task": "Write a report on X"}

    # (a) Real blueprint -> section appended, COMPLETE sentinel, progress updated.
    _reset_singletons()
    col = _bare_agent(Collector, "Collector", _mock_llm(["## Alpha\n\nProse body.", "new progress"]))
    spatial = {"d_2": {"role": "Data Analyst", "output": "- Claim: X is 5 GeV (p.3)"}}
    col._execute(task, spatial, {})
    rs = ReportState.instance()
    assert len(rs.additions) == 1 and rs.sections[0]["title"] == "Alpha"
    assert rs.task == "[SECTION_COMPLETE — ASSIGN NEXT SECTION]"
    assert rs.progress == "new progress"

    # (b) Absence-only blueprint -> no LLM call, nothing appended, SKIPPED sentinel.
    _reset_singletons()
    llm = _mock_llm(["should never be called"])
    col = _bare_agent(Collector, "Collector", llm)
    spatial = {"d_2": {"role": "Data Analyst", "output": "State Deficiency: density\nabsent: ratio"}}
    col._execute(task, spatial, {})
    rs = ReportState.instance()
    assert len(rs.additions) == 0
    assert rs.task == "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
    assert llm.gen.call_count == 0

    # (c) Sentinel output from the LLM itself -> nothing appended either.
    _reset_singletons()
    col = _bare_agent(Collector, "Collector", _mock_llm(["[NO NEW EVIDENCE]"]))
    spatial = {"d_2": {"role": "Data Analyst", "output": "- Claim: real content"}}
    col._execute(task, spatial, {})
    assert len(ReportState.instance().additions) == 0
    print("PASS  test_collector_append_and_skip")


def test_absence_and_sentinel_pins():
    assert _ABSENCE_RE.search("State Deficiency: melting point")
    assert _ABSENCE_RE.search("[NO NEW EVIDENCE]")
    assert _ABSENCE_RE.search("Term absent from the knowledge base")
    assert not _ABSENCE_RE.search("The melting point is 1500 K.")
    for sentinel in ("[REMOVE_SECTION]", "[NO_REVISION_NEEDED]", "[HOLD]", "[NO NEW EVIDENCE]"):
        assert sentinel in _SENTINEL_OUTPUTS
    print("PASS  test_absence_and_sentinel_pins")


# ---------------------------------------------------------------------------
# Stage 4.5 — prompt context-block rendering pins
# ---------------------------------------------------------------------------

def test_context_block_action_rendering():
    _reset_singletons()
    redacting = PromptSetRegistry.get("redacting")
    block = redacting.get_context_block("Data Analyst", action=7)
    assert "append" in block and "Current Action" in block
    assert redacting.get_context_block("Data Analyst", action=None) == ""
    assert redacting.get_context_block("Data Analyst") == ""

    # Handcrafted set: phase context renders; passing an action must not raise
    # (Stage 4.5 will ADD the action line here — do not assert its absence).
    handcrafted = PromptSetRegistry.get("handcrafted_redacting")
    block = handcrafted.get_context_block("Lead Architect")
    assert "PLANNING" in block and "Round" in block
    handcrafted.get_context_block("Lead Architect", action=3)  # must not raise
    print("PASS  test_context_block_action_rendering")


# ---------------------------------------------------------------------------
# Stage 4.6 — inference artifact path pin (meta-commentary filter)
# ---------------------------------------------------------------------------

def test_filter_meta_commentary():
    text = (
        "## Results\n"
        "The measured mass is 2.1 solar masses. The RAG results confirm this value.\n"
        "The next section will discuss implications.\n"
        "This conclusion stands on its own."
    )
    out = filter_meta_commentary(text)
    assert "## Results" in out
    assert "2.1 solar masses" in out                      # clean sentence kept
    assert "RAG results" not in out                       # meta sentence removed
    assert "next section" not in out                      # transition line removed
    assert "stands on its own" in out
    print("PASS  test_filter_meta_commentary")


# ---------------------------------------------------------------------------
# Stage 2.1 (Researcher path) — query parsing pin
# ---------------------------------------------------------------------------

def test_parse_queries():
    # Local import: pulls the RAG import chain (chromadb), keep isolated here.
    from qmix_report_writer.agents.researcher import _parse_queries

    raw = (
        "1. neutron star equation of state\n"
        "2) quark gluon plasma viscosity | [source: 2105.06979.pdf]\n"
        "NO_QUERY\n"
        "[RESEARCH_EXHAUSTED]\n"
        "This is a long natural language sentence that should be rejected "
        "because it has far too many words to be a query.\n"
        "ends with a period.\n"
        "hydrodynamic flow anisotropy\n"
        "fourth valid query line\n"
    )
    queries = _parse_queries(raw)
    assert queries[0] == "neutron star equation of state"
    assert queries[1] == "quark gluon plasma viscosity"     # citation suffix stripped
    assert len(queries) == 3                                # capped at 3
    assert all("period" not in q and "NO_QUERY" not in q for q in queries)
    print("PASS  test_parse_queries")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all():
    passed = failed = 0
    cases = [
        ("test_node_import_and_eval_standalone", test_node_import_and_eval_standalone),
        ("test_data_analyst_sync_async_equivalence", test_data_analyst_sync_async_equivalence),
        ("test_lead_architect_sync_async_equivalence_and_parse", test_lead_architect_sync_async_equivalence_and_parse),
        ("test_citation_overlap_tagging", test_citation_overlap_tagging),
        ("test_inline_reference_rewrite_and_orphans", test_inline_reference_rewrite_and_orphans),
        ("test_bibliography_build", test_bibliography_build),
        ("test_trace_round_schema", test_trace_round_schema),
        ("test_revalidation_sections_and_directive_extract", test_revalidation_sections_and_directive_extract),
        ("test_phase_sequence_invariants", test_phase_sequence_invariants),
        ("test_scheduler_temporal_heuristic", test_scheduler_temporal_heuristic),
        ("test_drafting_blueprint_usable", test_drafting_blueprint_usable),
        ("test_parse_section_titles", test_parse_section_titles),
        ("test_action_space_consistency", test_action_space_consistency),
        ("test_trainer_select_actions_shape", test_trainer_select_actions_shape),
        ("test_score_delta_semantics", test_score_delta_semantics),
        ("test_collector_append_and_skip", test_collector_append_and_skip),
        ("test_absence_and_sentinel_pins", test_absence_and_sentinel_pins),
        ("test_context_block_action_rendering", test_context_block_action_rendering),
        ("test_filter_meta_commentary", test_filter_meta_commentary),
        ("test_parse_queries", test_parse_queries),
    ]
    for name, fn in cases:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
