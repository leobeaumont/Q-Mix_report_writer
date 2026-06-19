"""
LIVE integration test for the section-deletion feature.

Unlike tests/test_remove_duplicate_section.py (which mocks the LLM and proves the
plumbing), this test drives a REAL Collector against the configured Ollama model
to answer the behavioural question the trace analysis left open:

    When a whole section is blatantly redundant and the directive says to remove
    it, does the live model actually emit `[REMOVE_SECTION]` so the pipeline
    deletes the section?

It seeds two sections where section_2 is a verbatim duplicate of section_1, puts
the pipeline into SECTION_REVIEW targeting the duplicate, and runs the Collector
end-to-end. Because LLM output is non-deterministic, it runs several trials and
reports how often the section was actually removed. On any miss it prints the
model's raw response so you can see what it did instead (e.g. rewrote the
section rather than deleting it).

Requirements: Ollama running locally with the model in configs/default.yaml.
The test SKIPS (does not fail) if Ollama is unreachable.

Run:  .venv/Scripts/python.exe tests/test_remove_section_live.py
"""

import asyncio
import sys
import urllib.request

sys.path.insert(0, ".")

from qmix_report_writer.utils.globals import ReportState, SourceBuffer
from qmix_report_writer.utils.config import get_llm_config
from qmix_report_writer.handcrafted_graph.state import PhaseState
from qmix_report_writer.handcrafted_graph.phases import PhaseType

TRIALS = 3

DUP_BODY = (
    "The pseudo-critical temperature of the QCD chiral crossover at vanishing "
    "baryon chemical potential is T_pc = 156.5 +/- 1.5 MeV, as determined by "
    "continuum-extrapolated lattice QCD using the chiral susceptibility."
)


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _reset_singletons():
    ReportState._instance = None
    SourceBuffer._instance = None
    PhaseState._instance = None


def _seed_duplicate_report():
    """Two sections; section_2 is a verbatim duplicate of section_1."""
    rs = ReportState.instance()
    rs.reset()
    rs.append(
        f"## Pseudo-Critical Temperature of the Chiral Crossover\n\n{DUP_BODY}",
        "progress stub",
    )
    rs.append(
        f"## Chiral Crossover Temperature (Duplicate)\n\n{DUP_BODY}",
        "progress stub",
    )


def _set_section_review_remove(section_idx: int):
    ps = PhaseState.instance()
    ps.set_phase(PhaseType.SECTION_REVIEW)
    rs = ReportState.instance()
    rs.review_section_idx = section_idx
    rs.validation_directive = (
        f"- section_2: This section is a verbatim duplicate of section_1 and adds "
        f"no new information. Remove this section entirely."
    )


async def _run_one_trial(trial: int):
    """Returns (removed: bool, raw_response: str)."""
    from qmix_report_writer.agents.collector import Collector  # real LLM via get_llm()

    _reset_singletons()
    _seed_duplicate_report()
    _set_section_review_remove(section_idx=1)  # target the duplicate -> section_2

    # llm_name="" would not resolve; pass the configured default model explicitly.
    col = Collector(llm_name=get_llm_config().get("default_model"))  # real Ollama LLM
    # The handcrafted graph reassigns every node's prompt set (graph.py:122-124).
    # Collector.__init__ defaults to "redacting", which does NOT inject the
    # REVISION DIRECTIVE / [REMOVE_SECTION] instruction — only "handcrafted_redacting"
    # does. Match the real pipeline or the model never sees the removal directive.
    from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry
    col.prompt_set = PromptSetRegistry.get("handcrafted_redacting")
    raw = await col._async_execute({"task": "Remove duplicated section"}, {}, {})

    rs = ReportState.instance()
    remaining = [s["id"] for s in rs.sections]
    removed = remaining == ["section_1"]
    return removed, str(raw), remaining


async def test_live_remove_duplicate_section():
    if not _ollama_up():
        print("SKIP  test_live_remove_duplicate_section: Ollama not reachable on :11434")
        return "skip"

    model = get_llm_config().get("default_model")
    print(f"Model under test: {model}")
    print(f"Running {TRIALS} trial(s) of the blatant-duplicate removal scenario...\n")

    removed_count = 0
    for t in range(1, TRIALS + 1):
        removed, raw, remaining = await _run_one_trial(t)
        first_line = raw.strip().splitlines()[0] if raw.strip() else "(empty)"
        if removed:
            removed_count += 1
            print(f"  trial {t}: REMOVED  ✓   (response: {first_line[:80]!r})")
        else:
            print(f"  trial {t}: kept     ✗   surviving={remaining}")
            print(f"            model response (first 200 chars): {raw.strip()[:200]!r}")

    print(f"\nResult: section removed in {removed_count}/{TRIALS} trials.")

    # The feature is considered working end-to-end if the live model triggers the
    # deletion path at least once. The ratio above tells you how reliable it is.
    assert removed_count >= 1, (
        "Live model NEVER emitted [REMOVE_SECTION] for a verbatim-duplicate section. "
        "The deletion plumbing works (see test_remove_duplicate_section.py), but the "
        "Collector prompt is not steering the model to choose removal — tune the "
        "directive / prompt wording."
    )
    print("PASS  test_live_remove_duplicate_section")
    return "pass"


async def _run_all():
    passed = failed = skipped = 0
    try:
        result = await test_live_remove_duplicate_section()
        if result == "skip":
            skipped += 1
        else:
            passed += 1
    except Exception as exc:
        print(f"FAIL  test_live_remove_duplicate_section: {exc}")
        import traceback; traceback.print_exc()
        failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped.")
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
