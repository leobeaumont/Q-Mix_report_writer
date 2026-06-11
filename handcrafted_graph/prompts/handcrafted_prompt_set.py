"""
Phase-aware prompt set for the handcrafted communication graph.

Registered under the key "handcrafted_redacting".

This prompt set wraps the existing redacting role descriptions and constraints,
then prepends a phase context block so that each agent knows:
  - Which stage of the pipeline it is in (PLANNING, RESEARCH, DRAFTING, …)
  - What its specific objective is for that stage
  - Who it is receiving from / sending to this round

Phase context is read dynamically from PhaseState.instance() so the same
prompt set instance works across all phases without re-instantiation.
"""

from __future__ import annotations

import re

from prompt.prompt_set import PromptSet
from prompt.prompt_set_registry import PromptSetRegistry
from prompt.redacting_prompt_set import (
    ROLE_DESCRIPTION,
    ROLE_CONSTRAINTS,
    JSON_SCHEMA,
)
from handcrafted_graph.phases import PhaseType


def _extract_section_directive(directive: str, section_id: str) -> str:
    """Return the instruction for section_id from a multi-section directive string.

    The directive is a bullet list of the form:
        - section_N: <instruction text>
        - section_M: <instruction text>
        ...
    Each instruction may span multiple lines via indentation.
    Returns the instruction text (stripped), or "" if section_id is not found.
    """
    pattern = rf'(?:^|\n)\s*-\s*{re.escape(section_id)}\s*:\s*(.*?)(?=\n\s*-\s*section_|\Z)'
    match = re.search(pattern, directive, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Phase context blocks injected into the system prompt
# ---------------------------------------------------------------------------

PHASE_CONTEXT: dict[PhaseType, str] = {
    PhaseType.PLANNING: (
        "### Pipeline Phase: PLANNING\n"
        "The team is establishing the overall structure and outline of the report. "
        "Focus on decomposing the task into sections before any writing begins. "
        "No prose should be written to the report yet — only structural decisions."
    ),
    PhaseType.RESEARCH: (
        "### Pipeline Phase: RESEARCH\n"
        "The team is gathering evidence from the knowledge base. "
        "Every factual claim must be grounded in RAG-retrieved documents. "
        "Do not draft prose — produce structured evidence atoms and synthesis only."
    ),
    PhaseType.DRAFTING: (
        "### Pipeline Phase: DRAFTING\n"
        "The team is writing the report section by section. "
        "Work on exactly one section per cycle. "
        "The Collector will produce the final prose; other agents supply structured input."
    ),
    PhaseType.SECTION_REVIEW: (
        "### Pipeline Phase: SECTION REVIEW\n"
        "The team is reviewing and revising one section at a time. "
        "Each agent focuses exclusively on the single section currently shown — "
        "do not comment on or modify any other section of the report."
    ),
    PhaseType.VALIDATION: (
        "### Pipeline Phase: VALIDATION\n"
        "The team is performing a final cross-section consistency check using a sliding window. "
        "Each pass covers a subset of adjacent sections. "
        "Flag ONLY serious structural problems: factual contradictions between sections, "
        "significant verbatim or near-verbatim content duplication, or transitions so abrupt "
        "that comprehension is genuinely impaired. "
        "Do NOT flag the mere absence of a bridging sentence or minor stylistic roughness — "
        "imperfect phrasing between sections is not a failure. "
        "Do NOT attempt per-section fact-checking against sources — no source chunks are "
        "available here; that was done in SECTION_REVIEW. "
        "Do NOT flag sections as missing or incomplete."
    ),
}

# Per-(phase, role) objective: what this agent should accomplish this phase.
PHASE_ROLE_OBJECTIVES: dict[tuple[PhaseType, str], str] = {
    # PLANNING
    (PhaseType.PLANNING, "Lead Architect"): (
        "Use the Researcher's coverage report to build a numbered section outline. "
        "ONLY plan sections for topics the Researcher explicitly confirmed are present "
        "in the knowledge base — do not invent sections around topics not mentioned. "
        "Each section entry must have a title and a one-sentence scope statement. "
        "TARGET SECTION COUNT: aim for 6–8 sections. Never exceed 10. "
        "If the knowledge base covers many sub-topics, merge related ones into broader "
        "sections rather than splitting them. Prefer depth over breadth — a focused "
        "8-section report is better than a shallow 15-section one. "
        "If the knowledge base covers very few topics, plan as few as 1–3 sections. "
        "CRITICAL: If the Researcher returned `[RESEARCH_EXHAUSTED]` or provided no "
        "confirmed topics, you have NO evidence — do NOT produce a section list. "
        "Output only: `[AWAITING_COVERAGE_DATA]` and nothing else. "
        "IMPORTANT: Round 1 of PLANNING only (and only if the Researcher provided coverage) "
        "— output the complete numbered section list as your entire output; "
        "this overrides the 'never list tasks' constraint. "
        "Round 2+ of PLANNING — the outline is already stored; do NOT rebuild it. "
        "In every PLANNING round, end by naming the first section target in one sentence."
    ),
    (PhaseType.PLANNING, "Researcher"): (
        "Perform a broad coverage scan of the knowledge base. "
        "Your goal is to answer: what topics, themes, and key concepts do the "
        "retrieved documents actually cover? "
        "Report a structured list of confirmed topics with brief descriptions. "
        "Completeness matters more than depth here — this output will be used "
        "to decide which sections to include in the report outline."
    ),
    # RESEARCH
    (PhaseType.RESEARCH, "Lead Architect"): (
        "Specify a focused search topic or question for the Researcher to query. "
        "Base your directive only on evidence already returned by previous RAG rounds. "
        "Do NOT invent document names, section numbers, equation numbers, or formulas "
        "that have not yet appeared in RAG results. "
        "If the Researcher reported 'State Deficiency', redirect to a related topic "
        "rather than repeating the failed query."
    ),
    (PhaseType.RESEARCH, "Researcher"): (
        "Execute a targeted RAG query using the LeadArchitect's directive. "
        "Return ALL relevant evidence atoms you found, with source attribution — "
        "do not withhold evidence you already have. "
        "Signal 'State Deficiency: [specific item]' only for sub-items that are "
        "completely absent from RAG results, after reporting what you did find. "
        "CRITICAL: If the retrieved documents contain no information relevant to "
        "the current directive (every document is off-topic or already fully "
        "reported), output `[RESEARCH_EXHAUSTED]` as the sole content of your "
        "response and nothing else."
    ),
    (PhaseType.RESEARCH, "Data Analyst"): (
        "Synthesise the Researcher's evidence atoms into a structured Markdown list "
        "of logical claims, data points, and gaps. This becomes the writing blueprint."
    ),
    # DRAFTING
    (PhaseType.DRAFTING, "Lead Architect"): (
        "You are in DRAFTING phase. The section to write has already been assigned — "
        "read it from 'Current Team Objective'.\n\n"
        "Your only job is to give DataAnalyst a focused, one-sentence evidence directive "
        "for that specific section, wrapped in <task>...</task> tags. "
        "Tell DataAnalyst what angle of evidence to structure (equations, parameters, "
        "experimental results, comparisons) to make the section substantive.\n\n"
        "Do NOT write prose. Do NOT pick a different section. Do NOT rebuild the outline."
    ),
    (PhaseType.DRAFTING, "Data Analyst"): (
        "Synthesize the Researcher's evidence from '### Received messages' into a dense "
        "structured Markdown list of claims, data points, and citations, scoped strictly "
        "to the section named in 'Current Team Objective'. "
        "CRITICAL: Do NOT generate specific numerical values (percentages, µs latencies, "
        "ratios, correlation coefficients, thresholds) unless they appear verbatim in "
        "the Researcher's message. If a metric is absent, write "
        "'State Deficiency: [metric name]' — never substitute a plausible-sounding number. "
        "CRITICAL: If the current team objective does not contain a real section title "
        "(e.g., it reads '[NEXT SECTION TO WRITE: ...]' but the title part is empty or "
        "missing), output only `[WAITING_FOR_DIRECTIVE]` and nothing else. "
        "CRITICAL: If the Researcher's message contains no evidence "
        "(only State Deficiency entries), output only `[NO NEW EVIDENCE]` and nothing else."
    ),
    (PhaseType.DRAFTING, "Researcher"): (
        "Query the knowledge base for evidence relevant to the section named in "
        "'Current Team Objective'. Return all retrieved evidence with source attribution. "
        "Signal 'State Deficiency: [item]' only for specific sub-items completely absent "
        "from the results. If no relevant documents are found, output `[RESEARCH_EXHAUSTED]`."
    ),
    (PhaseType.DRAFTING, "Collector"): (
        "Write the current report section as polished scientific prose using the "
        "DataAnalyst's structured input. Append it to the report. "
        "ALWAYS begin the section with a Markdown heading that matches the section "
        "title shown in 'Current Team Objective' (e.g., `## Section Title`). "
        "Write ONE new section only — do not reproduce content that already appears "
        "in the Previous Text Production. "
        "CRITICAL: End the section with a concluding statement about its own content. "
        "Do NOT write any sentence that references 'the next section', 'the following "
        "section', 'the subsequent section', or previews what comes next. Each section "
        "must stand alone — transitions between sections are forbidden. "
        "CRITICAL: If DataAnalyst marks any claim as 'State Deficiency', 'absent', "
        "or 'not found in RAG', omit that claim entirely — do not paraphrase, "
        "speculate, or rationalize around missing evidence. "
        "If DataAnalyst's entire output for this section consists only of State "
        "Deficiency entries with no supported claims, output nothing — do not write "
        "a gap-explanation paragraph. An empty section is better than meta-commentary. "
        "NEVER write about the retrieval process, the RAG system, data queries, "
        "or any pipeline-internal details — the report is a scientific document, "
        "not a process log."
    ),
    # SECTION_REVIEW — review round
    (PhaseType.SECTION_REVIEW, "Reviewer"): (
        "Audit ONLY the section shown under '### Section Content'. "
        "Do not comment on any other section of the report. "
        "Source chunks retrieved when this section was written are provided below the section text. "
        "Use them for fact-checking as follows:\n"
        "  - VERIFY: if a specific claim (number, named entity, technical term) appears in a source chunk, it is confirmed.\n"
        "  - IGNORE absence: if a claim is not present in the provided chunks, do NOT flag it as unsupported — "
        "those chunks cover only the drafting round; earlier research phases may have grounded it.\n"
        "  - FLAG contradictions: if a claim directly contradicts information in a provided chunk, report it.\n"
        "  - FLAG hallucinated specifics: if a highly specific data point (exact number, formula, proper name) "
        "appears nowhere in your entire context, flag it as potentially hallucinated.\n"
        "BE CONCISE: report only actionable issues — do not narrate the verification process or list "
        "confirmed claims. A short response is better than an exhaustive one.\n"
        "DECISION RULE — choose exactly one branch and follow it literally:\n"
        "  - If ZERO issues found → write the single token `[NO_REVISION_NEEDED]` and stop. Nothing else.\n"
        "  - If ANY issue found → list each issue as (1)/(2)/… with its correction and stop. "
        "Do NOT write `[NO_REVISION_NEEDED]` anywhere in your response when you have listed corrections.\n"
        "These two branches are mutually exclusive. Mixing them is an error."
    ),
    # SECTION_REVIEW — revision round
    (PhaseType.SECTION_REVIEW, "Data Analyst"): (
        "Prepare corrected content for the section currently under review. "
        "Output a structured Markdown list containing ONLY claims directly "
        "supported by RAG evidence — omit anything the Reviewer flagged as unverifiable. "
        "If the Reviewer's instruction is to remove this section entirely, "
        "output only `[REMOVE_SECTION]` and nothing else. "
        "If no RAG evidence supports any claim, output `[NO SUPPORTED CONTENT]` and nothing else."
    ),
    (PhaseType.SECTION_REVIEW, "Researcher"): (
        "Read DataAnalyst's message in '### Received messages'. "
        "If it contains one or more 'State Deficiency' entries, run a targeted "
        "RAG query for each flagged item and forward the retrieved evidence. "
        "If DataAnalyst's message contains no State Deficiency entries, "
        "output `[HOLD]` — do not add unsolicited evidence."
    ),
    (PhaseType.SECTION_REVIEW, "Collector"): (
        "Rewrite the section shown under '### Current Section' in full, starting with "
        "its Markdown heading, using DataAnalyst's corrected content as your source. "
        "Do NOT append a new section — produce a complete replacement of the shown text. "
        "If DataAnalyst's output is `[REMOVE_SECTION]`, output only `[REMOVE_SECTION]` "
        "and nothing else — the pipeline will delete the section. "
        "If DataAnalyst's output is `[NO SUPPORTED CONTENT]`, output nothing — "
        "leave the section unchanged rather than adding meta-commentary. "
        "NEVER output `[NO_REVISION_NEEDED]` — that signal belongs to the Reviewer only. "
        "Always produce either full section prose, `[REMOVE_SECTION]`, or nothing. "
        "Do NOT write any sentence that references 'the next section', 'the following "
        "section', 'the subsequent section', or previews future content. "
        "CRITICAL: Do NOT write about the retrieval process, the RAG system, the absence "
        "of data, or any pipeline-internal details."
    ),
    # VALIDATION — window review round
    (PhaseType.VALIDATION, "Reviewer"): (
        "Review ONLY the sections shown in your current window. "
        "Apply a VERY HIGH threshold — flag an issue ONLY if it meets the EXACT definition "
        "below. When in doubt, do NOT flag.\n\n"
        "  - Factual contradiction: two sections EXPLICITLY state the SAME specific quantity "
        "with CLEARLY DIFFERENT NUMBERS (e.g. Section_A says X = 5 GeV, Section_B says "
        "X = 10 GeV). The numerical values must be unmistakably incompatible.\n"
        "    NOT a contradiction: different sections citing different papers or fitting "
        "methods for the same result; one section saying a value comes from method A and "
        "another saying it comes from method B; different levels of precision; a section "
        "not mentioning a value another section discusses.\n\n"
        "  - Significant duplication: near-identical sentences or full paragraphs copied "
        "across sections — a reader could delete the block from one section and lose "
        "nothing. NOT duplication: same concept or result discussed from different angles.\n\n"
        "  - Severe transition: two adjacent sections discuss entirely unrelated topics "
        "with no logical connection — NOT merely an imperfect bridging sentence.\n\n"
        "Do NOT flag: different cited sources for the same claim, different derivation "
        "methods for the same value, absent bridging sentences, narrower scope in one "
        "section, imperfect phrasing, or anything a physicist would not find confusing. "
        "Do NOT attempt per-section fact-checking — no source chunks are available here.\n\n"
        "If no serious issues are found, state so in one sentence. "
        "In the synthesis round, consolidate all window notes into a concise global report "
        "covering only the serious issues identified above."
    ),
    (PhaseType.VALIDATION, "Lead Architect"): (
        "Receive the Reviewer's synthesised global quality notes and write a brief validation "
        "conclusion (2-4 sentences). Summarise the overall report quality.\n"
        "Apply a HIGH threshold for failure — output `[VALIDATION_FAILED]` ONLY if the "
        "Reviewer identified at least one serious problem: a factual contradiction between "
        "sections (same quantity, incompatible numbers) or a transition so incomprehensible "
        "that it genuinely impairs understanding. "
        "Imperfect phrasing, absent bridging sentences, shared technical vocabulary, repeated "
        "parameter values, and similar conclusions phrased differently are NOT failure criteria.\n"
        "If 'RE-VALIDATION MODE' is shown in context: follow its decision rule exactly — "
        "check only the listed prior issues and ignore anything else the Reviewer may have noted.\n"
        "End your response with EXACTLY one of these two tokens on its own line:\n"
        "  `[VALIDATION_PASSED]` — no serious cross-section issues remain.\n"
        "  `[VALIDATION_FAILED]` — at least one serious issue requires correction."
    ),
}


# ---------------------------------------------------------------------------
# Prompt set implementation
# ---------------------------------------------------------------------------

@PromptSetRegistry.register("handcrafted_redacting")
class HandcraftedPromptSet(PromptSet):
    """Phase-aware wrapper around the redacting prompt set."""

    def _current_phase(self) -> PhaseType:
        from handcrafted_graph.state import PhaseState
        return PhaseState.instance().current_phase

    def get_description(self, role: str) -> str:
        from utils.globals import ReportState
        base = ROLE_DESCRIPTION.get(role, "")
        phase = self._current_phase()

        # In re-validation mode the Reviewer is not a general auditor — replace the
        # VALIDATION phase context with a narrower RE-VALIDATION description so the
        # base system prompt does not prime it to hunt for new problems.
        if (phase == PhaseType.VALIDATION
                and role == "Reviewer"
                and ReportState.instance().validation_issues):
            phase_block = (
                "### Pipeline Phase: RE-VALIDATION\n"
                "The previous validation pass identified specific issues. "
                "Your role here is purely confirmatory: you are NOT asked to find new problems. "
                "You will be shown the exact issues that were flagged in the prior pass. "
                "Your sole task is to determine whether each of those issues has been "
                "corrected in the current text."
            )
        else:
            phase_block = PHASE_CONTEXT.get(phase, "")

        extra = f"\n\n{phase_block}" if phase_block else ""
        return base + extra

    def get_context_block(self, role: str, **kwargs) -> str:
        """Inject phase, round, per-role objective, and section list into the user prompt."""
        from handcrafted_graph.state import PhaseState
        state = PhaseState.instance()
        phase = state.current_phase
        round_n = state.round_in_phase
        objective = PHASE_ROLE_OBJECTIVES.get((phase, role), "")

        # In re-validation mode, replace the general VALIDATION objectives for Reviewer
        # and LeadArchitect so the model's primary instruction is compliance-checking,
        # not quality auditing.
        if phase == PhaseType.VALIDATION:
            from utils.globals import ReportState
            _prior = ReportState.instance().validation_issues
            if _prior:
                if role == "Reviewer":
                    objective = (
                        "You are in RE-VALIDATION MODE. Do NOT perform a general quality audit. "
                        "You will be shown the exact issues flagged in the previous validation pass. "
                        "For each issue relevant to the sections in your current window, state "
                        "RESOLVED or STILL PRESENT. Report nothing else."
                    )
                elif role == "Lead Architect":
                    objective = (
                        "You are in RE-VALIDATION MODE. Do NOT summarise overall report quality. "
                        "Your only task: using the Reviewer's compliance notes and the RE-VALIDATION "
                        "checklist below, decide whether every listed issue is now resolved. "
                        "Apply the decision rule exactly — ignore anything outside the checklist."
                    )

        block = "### Current Pipeline Context\n"
        block += f"**Phase:** {phase.value.upper()} | **Round:** {round_n}\n"
        if objective:
            block += f"**Your objective this round:** {objective}\n"

        # Inject section context so agents never have to guess or hallucinate IDs.
        if phase == PhaseType.SECTION_REVIEW:
            from utils.globals import ReportState
            report_state = ReportState.instance()
            idx = report_state.review_section_idx
            sections = report_state.sections
            if 0 <= idx < len(sections):
                section = sections[idx]
                block += (
                    f"\n**Current section under review:**\n"
                    f"  ID: `{section['id']}` | Title: {section['title'] or '(untitled)'} "
                    f"| Section {idx + 1} of {len(sections)}\n"
                )
            # Inject per-section actions from a previous failed validation pass.
            # Only the instruction for THIS section is shown — the full multi-section
            # directive must never be exposed, as the Reviewer will otherwise apply
            # another section's instructions to the current one.
            directive = report_state.validation_directive
            if directive and 0 <= idx < len(sections):
                current_id = sections[idx]["id"]
                section_instruction = _extract_section_directive(directive, current_id)
                if section_instruction:
                    if role == "Collector":
                        # Directive-bypass mode: Collector applies the change directly.
                        # Give it writer instructions, not reviewer audit instructions.
                        block += (
                            f"\n**REVISION DIRECTIVE for `{current_id}` — apply this change:**\n"
                            f"{section_instruction}\n\n"
                            f"Make ONLY the change described above. Do NOT rewrite the section from "
                            f"scratch — read the existing section content and apply the minimal edit "
                            f"required. Then output the complete revised section starting with its "
                            f"Markdown heading.\n"
                            f"If the directive instructs you to remove or delete this section entirely, "
                            f"output exactly `[REMOVE_SECTION]` and nothing else.\n"
                        )
                    else:
                        block += (
                            f"\n**REVISION DIRECTIVE for `{current_id}` — this is your ONLY task:**\n"
                            f"{section_instruction}\n\n"
                            f"**STRICT SCOPE:** Do NOT fact-check any other claims in this section. "
                            f"Do NOT flag issues not listed in the directive above. "
                            f"Do NOT introduce new corrections — the directive is the complete and "
                            f"authoritative list of what must change. "
                            f"Check ONLY whether the directive has been applied: if not yet applied, "
                            f"output the remaining items as numbered corrections; "
                            f"if fully applied, output [NO_REVISION_NEEDED] and nothing else.\n"
                        )
                else:
                    block += (
                        f"\n**Validation note:** `{current_id}` was NOT flagged for any changes "
                        f"in the previous validation pass. **Output [NO_REVISION_NEEDED] "
                        f"immediately. Do not perform fact-checking or review of this section.**\n"
                    )

        if phase == PhaseType.VALIDATION:
            from utils.globals import ReportState
            report_state = ReportState.instance()
            window_info = report_state.validation_window
            prior_issues = report_state.validation_issues
            if window_info is not None:
                # Window review: show scope so agents don't flag out-of-window content
                i, n_windows, window_sections = window_info
                ids = ", ".join(s["id"] for s in window_sections)
                block += (
                    f"\n**Validation window {i + 1} of {n_windows}** — "
                    f"reviewing: {ids}. "
                    f"All other sections are outside this window — do not reference them.\n"
                )
                # In re-validation mode, give the Reviewer only the prior checklist.
                # It must not flag anything new — only report resolved / still-present.
                if prior_issues and role == "Reviewer":
                    current_ids = ", ".join(s["id"] for s in report_state.sections)
                    block += (
                        f"\n**RE-VALIDATION CHECKLIST — prior issues to verify:**\n"
                        f"**Sections still in the report:** {current_ids}\n"
                        f"(Any section not in this list has been removed as part of the fix.)\n\n"
                        f"{prior_issues[:1200]}\n\n"
                        f"For each issue above that involves the sections in this window ({ids}), "
                        f"state RESOLVED or STILL PRESENT. "
                        f"If an issue references a section that has been removed from the report, "
                        f"mark it as RESOLVED. "
                        f"Do NOT flag any new issues. Do NOT comment on anything not in this list.\n"
                    )
            else:
                # Synthesis: provide section list + all accumulated window notes
                block += f"\n**Full report structure:**\n"
                block += report_state.list_sections(verbose=True) + "\n"
                notes = report_state.validation_notes
                if notes:
                    block += f"\n**Window review notes ({len(notes)} window(s) reviewed):**\n"
                    for j, note in enumerate(notes):
                        excerpt = note[:800] + ("…" if len(note) > 800 else "")
                        block += f"\n--- Window {j + 1} ---\n{excerpt}\n"
                # Re-validation synthesis: the LA decides solely from the prior checklist.
                if prior_issues:
                    block += (
                        f"\n**RE-VALIDATION MODE — compliance check only.**\n"
                        f"The following issues were identified in the previous validation pass:\n"
                        f"{prior_issues[:1200]}\n\n"
                        f"Decision rule:\n"
                        f"  → If ALL listed issues are resolved: output `[VALIDATION_PASSED]`.\n"
                        f"  → If at least one listed issue is still present: output `[VALIDATION_FAILED]`.\n"
                        f"Do NOT introduce new failure criteria. "
                        f"Do NOT fail on anything not in the list above. "
                        f"The window notes above reflect compliance checks only — "
                        f"any new findings in those notes must be ignored.\n"
                        f"Output ONLY the verdict sentinel on its own line — "
                        f"no directive, no task, no explanation.\n"
                    )

        return block

    @staticmethod
    def get_constraint(role: str) -> str:
        return ROLE_CONSTRAINTS.get(role, ROLE_CONSTRAINTS.get("Technical Writer", ""))

    def get_schema(self, role: str) -> dict:
        return JSON_SCHEMA.get(role, {})

    # ------------------------------------------------------------------
    # Unused abstract method stubs (required by PromptSet ABC)
    # ------------------------------------------------------------------

    @staticmethod
    def get_role() -> str:
        return ""

    def get_role_connection(self):
        return []

    @staticmethod
    def get_format():
        return None

    @staticmethod
    def get_answer_prompt(question: str, role: str = "") -> str:
        return question

    @staticmethod
    def get_decision_constraint() -> str:
        return ""

    @staticmethod
    def get_decision_role() -> str:
        return ""
