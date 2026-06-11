"""
Targeted test for the section-deletion feature in the realistic case the
pipeline actually hits: a report with SEVERAL sections, one of which is a
duplicate, and the revision directive asks to remove that duplicate.

The single-section happy path is already covered by
tests/test_directive_review.py::test_directive_remove_section. What that test
cannot catch is a WRONG-section removal — section targeting is resolved by
positional index (review_section_idx) while section IDs are assigned at append
time and never renumbered. This test seeds three sections, removes the
duplicated middle one, and asserts that the correct section disappeared and the
others survived intact.
"""

import asyncio
import sys

sys.path.insert(0, ".")

from unittest.mock import AsyncMock, MagicMock

from utils.globals import ReportState, SourceBuffer
from handcrafted_graph.state import PhaseState
from handcrafted_graph.phases import PhaseType
from agents.collector import Collector


# ---------------------------------------------------------------------------
# Helpers (kept local so the file runs standalone)
# ---------------------------------------------------------------------------

def _reset_singletons():
    ReportState._instance = None
    SourceBuffer._instance = None
    PhaseState._instance = None


def _seed_sections(sections: list[tuple[str, str]]) -> list[str]:
    """Seed a fresh report with (title, body) pairs. Returns the section ids."""
    rs = ReportState.instance()
    rs.reset()
    ids = []
    for title, body in sections:
        ids.append(rs.append(f"## {title}\n\n{body}", "progress stub"))
    return ids


def _make_collector(llm_responses: list[str]) -> Collector:
    """Collector whose mocked LLM returns each item of llm_responses in turn."""
    col = Collector.__new__(Collector)
    col.report = ReportState.instance()
    col.source_buffer = SourceBuffer.instance()

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
    ps = PhaseState.instance()
    ps.set_phase(PhaseType.SECTION_REVIEW)
    rs = ReportState.instance()
    rs.review_section_idx = section_idx
    rs.validation_directive = directive


# ---------------------------------------------------------------------------
# Test 1 — remove the duplicated middle section, keep the rest
# ---------------------------------------------------------------------------

async def test_remove_duplicate_middle_section():
    _reset_singletons()
    dup_body = "The pseudo-critical temperature is T_pc ~ 156 MeV at zero density."
    ids = _seed_sections([
        ("Chiral Transition",        dup_body),                       # section_1
        ("Chiral Transition (dup)",  dup_body),                       # section_2  <- duplicate
        ("Transport Properties",     "Shear viscosity over entropy is small."),  # section_3
    ])
    assert ids == ["section_1", "section_2", "section_3"], ids

    # Pipeline targets the duplicate by its positional index (1 -> section_2).
    _set_section_review(
        section_idx=1,
        directive=f"- {ids[1]}: This section duplicates section_1. Remove it entirely.",
    )

    col = _make_collector(["[REMOVE_SECTION]", "progress after removal"])
    await col._async_execute({"task": "dedupe"}, {}, {})

    rs = ReportState.instance()
    remaining_ids = [s["id"] for s in rs.sections]

    assert len(rs.sections) == 2, (
        f"Expected 2 sections after removal, got {len(rs.sections)}: {remaining_ids}"
    )
    # The duplicate (section_2) must be gone; IDs are NOT renumbered on removal.
    assert remaining_ids == ["section_1", "section_3"], (
        f"Wrong section removed. Surviving ids: {remaining_ids} (expected section_1, section_3)"
    )
    # Surviving content must be untouched and in order.
    assert rs.sections[0]["content"].startswith("## Chiral Transition\n")
    assert rs.sections[1]["content"].startswith("## Transport Properties\n")
    # self.content must be rebuilt without the removed section.
    assert "Chiral Transition (dup)" not in rs.content
    assert rs.content.count(dup_body) == 1, "Duplicate body should appear exactly once now."
    print("PASS  test_remove_duplicate_middle_section")


# ---------------------------------------------------------------------------
# Test 2 — out-of-range target resolves to no id -> nothing is removed
# ---------------------------------------------------------------------------

async def test_remove_out_of_range_index_is_safe():
    """If review_section_idx can't resolve to a section, the report is left intact."""
    _reset_singletons()
    ids = _seed_sections([
        ("Alpha", "First body."),
        ("Beta",  "Second body."),
    ])
    _set_section_review(
        section_idx=5,  # out of range -> _extract_section_id_from_review_index() returns ""
        directive=f"- {ids[0]}: Remove this section entirely.",
    )

    col = _make_collector(["[REMOVE_SECTION]", "progress stub"])
    await col._async_execute({"task": "dedupe"}, {}, {})

    rs = ReportState.instance()
    assert [s["id"] for s in rs.sections] == ["section_1", "section_2"], (
        f"No section should have been removed, but got {[s['id'] for s in rs.sections]}"
    )
    print("PASS  test_remove_out_of_range_index_is_safe")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all():
    passed = failed = 0
    for name, coro in [
        ("test_remove_duplicate_middle_section", test_remove_duplicate_middle_section()),
        ("test_remove_out_of_range_index_is_safe", test_remove_out_of_range_index_is_safe()),
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
