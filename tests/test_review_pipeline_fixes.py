"""
Regression tests for the SECTION_REVIEW / VALIDATION pipeline fixes:

  1. [REMOVE_SECTION] without provenance (no directive, no Reviewer removal
     instruction) is refused — the section survives.
  2. [REMOVE_SECTION] backed by ReportState.removal_authorized is honored.
  3. _REMOVAL_REQUEST_RE recognises explicit removal instructions in critiques
     and does not fire on ordinary corrections.
  4. _build_section_windows never emits a trailing window made only of
     already-covered sections, while still covering every adjacent pair.
  5. _trim_truncated_tail cuts a max_tokens-truncated response back to the last
     complete sentence (and leaves boundary-free text alone).
"""

import asyncio
import sys

sys.path.insert(0, ".")

from unittest.mock import AsyncMock, MagicMock

from utils.globals import ReportState, SourceBuffer
from handcrafted_graph.state import PhaseState
from handcrafted_graph.phases import PhaseType
from handcrafted_graph.graph import HandcraftedGraph, _REMOVAL_REQUEST_RE
from agents.collector import Collector
from llm.ollama_chat import _trim_truncated_tail


# ---------------------------------------------------------------------------
# Helpers (kept local so the file runs standalone)
# ---------------------------------------------------------------------------

def _reset_singletons():
    ReportState._instance = None
    SourceBuffer._instance = None
    PhaseState._instance = None


def _seed_sections(sections: list[tuple[str, str]]) -> list[str]:
    rs = ReportState.instance()
    rs.reset()
    return [rs.append(f"## {t}\n\n{b}", "progress stub") for t, b in sections]


def _make_collector(llm_responses: list[str]) -> Collector:
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


def _enter_section_review(section_idx: int):
    PhaseState.instance().set_phase(PhaseType.SECTION_REVIEW)
    rs = ReportState.instance()
    rs.review_section_idx = section_idx
    rs.validation_directive = ""  # normal review pass, not directive-bypass


# DataAnalyst relay so the Collector's spatial_info guard lets it execute.
_DA_REMOVE_MSG = {"da_0": {"role": "Data Analyst", "output": "[REMOVE_SECTION]"}}


# ---------------------------------------------------------------------------
# Test 1 — unauthorized [REMOVE_SECTION] is refused
# ---------------------------------------------------------------------------

async def test_unauthorized_remove_is_refused():
    _reset_singletons()
    _seed_sections([("Alpha", "First body."), ("Beta", "Second body.")])
    _enter_section_review(section_idx=0)
    ReportState.instance().removal_authorized = False

    col = _make_collector(["[REMOVE_SECTION]"])
    await col._async_execute({"task": "review"}, dict(_DA_REMOVE_MSG), {})

    rs = ReportState.instance()
    assert [s["id"] for s in rs.sections] == ["section_1", "section_2"], (
        f"Unauthorized removal went through — surviving: {[s['id'] for s in rs.sections]}"
    )
    print("PASS  test_unauthorized_remove_is_refused")


# ---------------------------------------------------------------------------
# Test 2 — removal_authorized flag allows the removal
# ---------------------------------------------------------------------------

async def test_authorized_remove_goes_through():
    _reset_singletons()
    _seed_sections([("Alpha", "First body."), ("Beta", "Second body.")])
    _enter_section_review(section_idx=0)
    ReportState.instance().removal_authorized = True

    col = _make_collector(["[REMOVE_SECTION]"])
    await col._async_execute({"task": "review"}, dict(_DA_REMOVE_MSG), {})

    rs = ReportState.instance()
    assert [s["id"] for s in rs.sections] == ["section_2"], (
        f"Authorized removal failed — surviving: {[s['id'] for s in rs.sections]}"
    )
    print("PASS  test_authorized_remove_goes_through")


# ---------------------------------------------------------------------------
# Test 3 — removal-request detection in Reviewer critiques
# ---------------------------------------------------------------------------

def test_removal_request_regex():
    positive = [
        "This section duplicates section_1. Remove this section entirely.",
        "(1) The whole section should be deleted as redundant.",
        "Recommend removal of the section due to total overlap.",
        "Drop the section; its content is covered elsewhere.",
    ]
    negative = [
        "(1) Remove the claim about 70 keV — it is unsupported.",
        "(2) The formula is incorrect; replace it with the source's version.",
        "The transition between paragraphs is abrupt.",
        "(3) Delete the sentence asserting RG invariance.",
    ]
    for text in positive:
        assert _REMOVAL_REQUEST_RE.search(text), f"Should authorize removal: {text!r}"
    for text in negative:
        assert not _REMOVAL_REQUEST_RE.search(text), f"Should NOT authorize removal: {text!r}"
    print("PASS  test_removal_request_regex")


# ---------------------------------------------------------------------------
# Test 4 — no trailing single-section validation window
# ---------------------------------------------------------------------------

def test_no_trailing_singleton_window():
    sections = [{"id": f"section_{i}", "content": "x" * 3800} for i in range(1, 5)]
    windows = HandcraftedGraph._build_section_windows(
        sections, window_size=6000, overlap_sections=1
    )
    ids = [[s["id"] for s in w] for w in windows]

    # min-2 rule -> pairwise windows; the trailing [section_4]-only window
    # produced by the old overlap advance must be gone.
    assert ids == [
        ["section_1", "section_2"],
        ["section_2", "section_3"],
        ["section_3", "section_4"],
    ], f"Unexpected windows: {ids}"

    # Single-section reports still get exactly one window.
    one = HandcraftedGraph._build_section_windows(sections[:1], 6000, 1)
    assert [[s["id"] for s in w] for w in one] == [["section_1"]]
    print("PASS  test_no_trailing_singleton_window")


# ---------------------------------------------------------------------------
# Test 5 — truncated-tail trimming
# ---------------------------------------------------------------------------

def test_trim_truncated_tail():
    truncated = (
        "(1) The claim is unsupported by the cited source.\n"
        "(2) The formula misstates the sign convention.\n"
        "(3) The intermediate-range potential is mis"
    )
    trimmed = _trim_truncated_tail(truncated)
    assert trimmed.endswith("misstates the sign convention."), f"Got: {trimmed!r}"
    assert "(3)" not in trimmed

    # No sentence boundary in the second half -> returned unchanged.
    no_boundary = "word " * 50
    assert _trim_truncated_tail(no_boundary) == no_boundary
    print("PASS  test_trim_truncated_tail")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all():
    passed = failed = 0
    cases = [
        ("test_unauthorized_remove_is_refused", test_unauthorized_remove_is_refused),
        ("test_authorized_remove_goes_through", test_authorized_remove_goes_through),
        ("test_removal_request_regex", test_removal_request_regex),
        ("test_no_trailing_singleton_window", test_no_trailing_singleton_window),
        ("test_trim_truncated_tail", test_trim_truncated_tail),
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
