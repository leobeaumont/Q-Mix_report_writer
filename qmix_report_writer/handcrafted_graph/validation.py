"""
Validation-loop helpers — window building, re-validation scoping, directive
decomposition.

Extracted from HandcraftedGraph (upgrade-plan Stage 2.4) as pure code motion:
these functions carry no graph state beyond the ReportState / PhaseConfig they
receive, and keeping them here leaves graph.py to its phase-execution core.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger("handcrafted_graph.validation")


def build_section_windows(
    sections: List[dict],
    window_size: int = 6000,
    overlap_sections: int = 1,
) -> List[List[dict]]:
    """Group sections into overlapping windows that fit within window_size chars.

    Each window contains complete sections (no mid-section cuts), preserving
    semantic boundaries. Consecutive windows share overlap_sections sections
    so cross-boundary transitions are always visible in at least one window.

    A minimum of 2 sections per window is enforced (unless fewer than 2 sections
    remain) so that adjacent sections always appear together and the Reviewer can
    detect cross-section duplication even when individual sections are large.
    """
    if not sections:
        return []
    windows: List[List[dict]] = []
    i = 0
    while i < len(sections):
        window: List[dict] = []
        total = 0
        j = i
        while j < len(sections):
            section_len = len(sections[j]["content"])
            # Enforce window_size limit only once we have ≥ 2 sections, so that
            # a single oversized section never fills the window alone.
            if total + section_len > window_size and len(window) >= 2:
                break
            window.append(sections[j])
            total += section_len
            j += 1
        windows.append(window)
        # Stop once a window reaches the final section: the overlap advance
        # would otherwise produce a trailing window containing only sections
        # already covered (e.g. a redundant single-section window).
        if j >= len(sections):
            break
        i += max(1, len(window) - overlap_sections)
    return windows


def revalidation_sections(report_state) -> List[dict]:
    """Sections to re-audit: every section named in the prior issues or directive.

    For a cross-section contradiction the first-pass issue text names both
    sections (e.g. "section_2 ... contradicts section_3 ...") and the
    decomposed directive carries a per-section entry for each, so the union of
    IDs parsed from both covers all sides of every issue. Returned in report
    order. Falls back to the full report if no section IDs can be parsed.
    """
    text = f"{report_state.validation_directive or ''}\n{report_state.validation_issues or ''}"
    nums = re.findall(r'section[_ ]?(\d+)', text, re.IGNORECASE)
    ids = {f"section_{n}" for n in nums}
    if not ids:
        return list(report_state.sections)
    named = [s for s in report_state.sections if s["id"] in ids]
    return named or list(report_state.sections)


def validation_windows(report_state, phase) -> List[List[dict]]:
    """Group sections into review windows for the VALIDATION phase.

    First pass: sliding overlapping windows bounded by window_size.
    Re-validation pass (validation_issues set): a SINGLE window containing
    every section referenced by the prior issues/directive, so a cross-section
    fix is always verified with both sides visible at once. The sliding
    windows cannot do this when the two contradicting sections fall in
    different windows — a partial-view window can only guess, and a single
    false "STILL PRESENT" vote fails the whole pass.
    """
    if report_state.validation_issues:
        rv = revalidation_sections(report_state)
        return [rv] if rv else []
    return build_section_windows(
        report_state.sections, phase.window_size, phase.window_overlap_sections
    )


async def decompose_validation_directive(
    combined_issues: str,
    report_state,
    llm,
) -> str:
    """Ask the LLM to decompose global validation issues into per-section actions.

    Returns a bulleted list of `- section_N: <action>` items, or the raw
    combined_issues string if the LLM call fails or no LLM is available.
    """
    if llm is None:
        return combined_issues

    system = (
        "You are a report quality coordinator. A validation review found cross-section "
        "issues in a multi-section scientific report. Decompose each issue into concrete, "
        "unambiguous revision instructions.\n\n"
        "CRITICAL RULES:\n"
        "1. For factual contradictions where the SAME value is stated differently in "
        "multiple sections: pick ONE authoritative value (prefer the one cited with a "
        "specific source page) and list EVERY section that must be updated to use it. "
        "Never say 'reconcile' or 'align' — always give the exact value to use.\n"
        "2. For content duplication: name the section to keep and the section to shorten. "
        "Give the shortening section a specific UNIQUE angle to retain so it cannot end up "
        "saying the same thing as the section it is being differentiated from.\n"
        "3. For severe transitions: name which section's opening or closing sentence to revise.\n"
        "4. CRITICAL — `section_N` labels are INTERNAL identifiers the reader never sees. "
        "No section identifier (section_1, section_2, ...) may appear ANYWHERE in your "
        "output: not in the instruction, and not inside any replacement text you quote. "
        "This is the most common failure on repetition fixes — do NOT condense a duplicate "
        "by cross-referencing another section, e.g. 'established in section_2', 'derived in "
        "section_3', 'the parametrization from section_4', or 'as in section_6'. Instead "
        "either state the point self-containedly in condensed form, or refer to it by its "
        "topic in plain words (e.g. 'as established for even-even nuclei'). Each instruction "
        "must stand alone and reference only physical facts, observational evidence, or "
        "source citations — never another section."
    )
    section_list = report_state.list_sections(verbose=True)
    user = (
        f"### Report sections\n{section_list}\n\n"
        f"### Identified issues\n{combined_issues}\n\n"
        "Output ONLY a bulleted list using this exact format:\n"
        "  - <section_id>: <specific action with exact value if applicable>\n\n"
        "One bullet per section that needs changing. The section_id is used ONLY as the "
        "bullet label — never write it inside the action text or inside any quoted "
        "replacement prose (see rule 4). Any text you put in quotes will be inserted "
        "verbatim into the report, so it must read as self-contained prose. "
        "If the same factual value must appear in "
        "multiple sections, list each section separately and give EACH a distinct angle or "
        "sub-topic so they do not duplicate each other. "
        "Use exact section IDs from the list above for the bullet labels only. "
        "Skip praise or general observations."
    )
    message = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        response = await llm.agen(message, calling_agent="LeadArchitect")
        return response.strip() if response else combined_issues
    except Exception as exc:
        logger.warning(f"Directive decomposition failed: {exc}")
        return combined_issues
