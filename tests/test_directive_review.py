"""
Tests for the directive-bypass SECTION_REVIEW path:
  1. Collector applies a directive with empty spatial_info (no Reviewer/DataAnalyst)
  2. Collector removes a section when the directive says to remove it
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from utils.globals import ReportState, SourceBuffer
from handcrafted_graph.state import PhaseState
from handcrafted_graph.phases import PhaseType
from agents.collector import Collector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_singletons():
    ReportState._instance = None
    SourceBuffer._instance = None
    PhaseState._instance = None


def _seed_report(content: str, title: str = "Test Section") -> str:
    """Add one section to a fresh ReportState and return its id."""
    rs = ReportState.instance()
    rs.reset()
    section_id = rs.append(f"## {title}\n\n{content}", "progress stub")
    return section_id


def _make_collector(llm_responses: list[str]) -> Collector:
    """Return a Collector whose LLM always returns the next item from llm_responses."""
    col = Collector.__new__(Collector)
    col.report = ReportState.instance()
    col.source_buffer = SourceBuffer.instance()

    # Pull prompt_set from the real registry so context-block injection works.
    from prompt.prompt_set_registry import PromptSetRegistry
    col.prompt_set = PromptSetRegistry.get("redacting")
    col.role = "Collector"

    responses = iter(llm_responses)
    mock_llm = MagicMock()
    mock_llm.gen = MagicMock(side_effect=lambda *a, **kw: next(responses))
    mock_llm.agen = AsyncMock(side_effect=lambda *a, **kw: next(responses))
    col.llm = mock_llm
    return col


def _set_section_review(section_idx: int, directive: str):
    """Put the pipeline into SECTION_REVIEW with a given directive."""
    ps = PhaseState.instance()
    ps.set_phase(PhaseType.SECTION_REVIEW)
    rs = ReportState.instance()
    rs.review_section_idx = section_idx
    rs.validation_directive = directive


# ---------------------------------------------------------------------------
# Test 1 — directive application with empty spatial_info
# ---------------------------------------------------------------------------

async def test_directive_apply():
    _reset_singletons()
    section_id = _seed_report("Original content that needs fixing.")
    _set_section_review(
        section_idx=0,
        directive=f"- {section_id}: Rewrite the opening sentence to clarify the main claim.",
    )

    revised_text = "## Test Section\n\nRevised opening sentence with clarified claim."
    col = _make_collector([revised_text, "progress after revision"])

    # Empty spatial_info — the directive-bypass path
    await col._async_execute({"task": "Test task"}, {}, {})

    rs = ReportState.instance()
    assert rs.sections, "Section list must not be empty after revision."
    assert rs.sections[0]["content"] == revised_text, (
        f"Section content was not updated.\n"
        f"Expected: {revised_text!r}\n"
        f"Got:      {rs.sections[0]['content']!r}"
    )
    print("PASS  test_directive_apply")


# ---------------------------------------------------------------------------
# Test 2 — section removal via [REMOVE_SECTION] in directive-bypass
# ---------------------------------------------------------------------------

async def test_directive_remove_section():
    _reset_singletons()
    section_id = _seed_report("Content of the section that should be removed.")
    _set_section_review(
        section_idx=0,
        directive=f"- {section_id}: Remove this section entirely.",
    )

    col = _make_collector(["[REMOVE_SECTION]", "progress after removal"])

    await col._async_execute({"task": "Test task"}, {}, {})

    rs = ReportState.instance()
    assert len(rs.sections) == 0, (
        f"Section should have been removed but {len(rs.sections)} section(s) remain."
    )
    print("PASS  test_directive_remove_section")


# ---------------------------------------------------------------------------
# Test 3 — sections without a directive are skipped (spatial_info guard)
# ---------------------------------------------------------------------------

async def test_no_directive_no_execute():
    """Collector must stay idle when spatial_info is empty and no directive is set."""
    _reset_singletons()
    _seed_report("Content that must not change.")

    ps = PhaseState.instance()
    ps.set_phase(PhaseType.SECTION_REVIEW)
    rs = ReportState.instance()
    rs.review_section_idx = 0
    rs.validation_directive = ""  # no directive

    # Capture the full stored content (seed prepends the markdown heading)
    stored_before = rs.sections[0]["content"]

    col = _make_collector(["Should never be called"])

    await col._async_execute({"task": "Test task"}, {}, {})

    assert rs.sections[0]["content"] == stored_before, (
        "Section content changed even though no directive was set and spatial_info was empty."
    )
    print("PASS  test_no_directive_no_execute")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all():
    passed = failed = 0
    for name, coro in [
        ("test_directive_apply",       test_directive_apply()),
        ("test_directive_remove_section", test_directive_remove_section()),
        ("test_no_directive_no_execute",  test_no_directive_no_execute()),
    ]:
        try:
            await coro
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
