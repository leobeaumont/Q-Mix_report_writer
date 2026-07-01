"""
Integration tests for the PBDS enrichment wired into the Researcher.

Covers the contract requested for the wiring:
  * tool OFF -> the dedicated prompt section does not render (pipeline unchanged);
  * tool ON  -> a dedicated "Parameter Dependency Analysis" section renders with
                the connected parameters;
  * RESEARCH -> the per-run frontier prevents a parameter that was already
                surfaced from re-triggering the tool (no dependency-graph walking
                when its documents come back later);
  * DRAFTING -> annotates relationships for the section and ignores the frontier.

The Researcher's real __init__ builds a RAGManager/LLM, so these tests construct a
bare instance via __new__ and inject a fixture PBDSManager + NodeMatcher (the same
approach as tests/test_pbds_pipeline.py). No Ollama and no workbook are needed; the
LLM verifier is a fake returning fixed node ids.

Run:  .venv/Scripts/python.exe tests/test_pbds_researcher_integration.py
"""

import asyncio
import json
import sys

sys.path.insert(0, ".")

import networkx as nx

from qmix_report_writer.agents.researcher import Researcher
from qmix_report_writer.tools.pbds import NodeMatcher, PBDSManager
from qmix_report_writer.utils.globals import ReportState
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import PhaseType

# Same fixture graph as tests/test_pbds_pipeline.py.
_DESCRIPTIONS = {
    "Pellet_diameter_F8": "Pellet diameter",
    "Max_fuel_temperature_F10": "Max fuel temperature",
    "Cladding_thickness_F15": "Cladding thickness",
    "Core_power_F26": "Core power",
    "Core_inlet_temperature_F28": "Core inlet temperature",
}
_EDGES = [
    ("Pellet_diameter_F8", "Max_fuel_temperature_F10", "=F(C8)"),
    ("Core_power_F26", "Max_fuel_temperature_F10", "=G(C26)"),
    ("Core_inlet_temperature_F28", "Core_power_F26", "=H(C28)"),
    ("Max_fuel_temperature_F10", "Cladding_thickness_F15", "=I(C10)"),
]


class FakeLLM:
    """Verifier stub: agen() returns a fixed node_ids selection as JSON."""

    def __init__(self, node_ids):
        self._ids = node_ids

    async def agen(self, messages, response_schema=None, calling_agent=None):
        return json.dumps({"node_ids": self._ids, "reason": "test"})


def _fixture_manager():
    g = nx.DiGraph()
    for name in _DESCRIPTIONS:
        g.add_node(name, owner="MGR", row=int(name.rsplit("_F", 1)[1]), row_type="calculated_parameter")
    for src, dst, formula in _EDGES:
        g.add_edge(src, dst, formulas=[formula])
    m = PBDSManager("<in-memory>", default_k=1)
    m._graph = g
    m._descriptions = dict(_DESCRIPTIONS)
    return m


def _make_researcher(matched_ids):
    """A bare Researcher with the PBDS tool injected (no heavy __init__)."""
    r = Researcher.__new__(Researcher)
    manager = _fixture_manager()
    r._pbds_manager = manager
    r._pbds_matcher = NodeMatcher(manager)
    r._pbds_surfaced_nodes = set()
    r.llm = FakeLLM(matched_ids)
    r.role = "Researcher"
    r.prompt_set = None                       # skip context block in _build_user_prompt
    r.report = ReportState.instance()
    return r


def _docs(text):
    return [{"content": text, "source": "doc.pdf", "id": "c1"}]


# ---------------------------------------------------------------------------

async def test_tool_off_renders_no_section():
    r = Researcher.__new__(Researcher)
    r._pbds_manager = None
    r._pbds_matcher = None
    r._pbds_surfaced_nodes = set()
    r.role = "Researcher"
    r.prompt_set = None
    r.report = ReportState.instance()

    block = await r._pbds_block(_docs("max fuel temperature"))
    assert block == "", block
    prompt = r._build_user_prompt(
        {"task": "subject"}, {}, {}, "Current report state", "progress", pbds_block=block
    )
    assert "Parameter Dependency Analysis" not in prompt
    print("PASS  test_tool_off_renders_no_section")


async def test_research_block_renders_and_frontier_blocks_recursion():
    PhaseState.instance().set_phase(PhaseType.RESEARCH)
    r = _make_researcher(["Max_fuel_temperature_F10"])

    block = await r._pbds_block(_docs("we control the max fuel temperature"))
    assert "Max fuel temperature" in block, block
    # sources (causes) and effects both surfaced
    assert "Pellet diameter" in block or "Core power" in block, block
    assert "Cladding thickness" in block, block

    # It renders as a dedicated section, not a low-priority agent message.
    prompt = r._build_user_prompt(
        {"task": "subject"}, {}, {}, "Current report state", "progress", pbds_block=block
    )
    assert "### Parameter Dependency Analysis" in prompt, prompt

    # Frontier now holds the matched node + its neighbours.
    assert "Max_fuel_temperature_F10" in r._pbds_surfaced_nodes
    assert "Cladding_thickness_F15" in r._pbds_surfaced_nodes

    # Documents retrieved LATER for the same (now-surfaced) parameter must NOT
    # re-trigger the tool -> empty block (no dependency-graph walking).
    block2 = await r._pbds_block(_docs("more about the max fuel temperature"))
    assert block2 == "", block2
    print("PASS  test_research_block_renders_and_frontier_blocks_recursion")


async def test_drafting_annotates_and_ignores_frontier():
    PhaseState.instance().set_phase(PhaseType.DRAFTING)
    r = _make_researcher(["Max_fuel_temperature_F10"])
    # Pre-seed the frontier: DRAFTING must still annotate (it ignores the frontier).
    r._pbds_surfaced_nodes.add("Max_fuel_temperature_F10")

    block = await r._pbds_block(_docs("this section covers the max fuel temperature"))
    assert "is computed from" in block, block
    assert "formula:" in block, block
    # A second call still annotates (frontier untouched in drafting).
    block2 = await r._pbds_block(_docs("the max fuel temperature again"))
    assert block2 != "", block2
    print("PASS  test_drafting_annotates_and_ignores_frontier")


async def test_non_pbds_phase_is_noop():
    PhaseState.instance().set_phase(PhaseType.VALIDATION)
    r = _make_researcher(["Max_fuel_temperature_F10"])
    block = await r._pbds_block(_docs("max fuel temperature"))
    assert block == "", block
    print("PASS  test_non_pbds_phase_is_noop")


async def _run_all():
    tests = [
        test_tool_off_renders_no_section,
        test_research_block_renders_and_frontier_blocks_recursion,
        test_drafting_annotates_and_ignores_frontier,
        test_non_pbds_phase_is_noop,
    ]
    passed = failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {test.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
