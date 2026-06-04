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

from prompt.prompt_set import PromptSet
from prompt.prompt_set_registry import PromptSetRegistry
from prompt.redacting_prompt_set import (
    ROLE_DESCRIPTION,
    ROLE_CONSTRAINTS,
    JSON_SCHEMA,
)
from handcrafted_graph.phases import PhaseType


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
        "The team is performing a final global quality check on the complete report. "
        "Focus exclusively on cross-section issues: flow, transitions between sections, "
        "duplicated content, and global coherence. Do not rewrite content directly."
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
        "IMPORTANT: Round 1 of PLANNING only — output the complete numbered section list "
        "as your entire output; this overrides the 'never list tasks' constraint. "
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
        "You are in DRAFTING phase. Do NOT rebuild the section outline — that was "
        "completed in PLANNING. Your only task each round is to assign ONE new section. "
        "Check the 'Sections written so far' list: any section present there is written "
        "and complete — '(untitled)' means no heading was extracted, not that it is empty. "
        "IMPORTANT: If the 'Current Team Objective' reads "
        "`[SECTION_COMPLETE — ASSIGN NEXT SECTION]`, the Collector just finished "
        "writing the previous section. That section must NOT be re-assigned. "
        "Look at 'Sections written so far', find the first topic not yet listed, "
        "and assign it as the new section. "
        "IMPORTANT: If the 'Current Team Objective' reads "
        "`[SECTION_SKIPPED — ASSIGN NEXT SECTION]`, the previous section had no "
        "supporting evidence in the knowledge base and was skipped. That topic "
        "must NOT be re-assigned. Move directly to the next unwritten topic. "
        "If no unwritten topics remain, output only `[DRAFTING_COMPLETE]`."
        "To find the next section: look at the progress summary for the topic sequence "
        "established so far, then assign the next logical topic not yet written. "
        "State the new section title and provide a one-sentence directive for DataAnalyst. "
        "Do not write prose yourself. "
        "Do NOT invent document names, section numbers, page numbers, equation numbers, "
        "or formulas that have not yet appeared in RAG results. "
        "Do NOT invent names of external frameworks, papers, or authors. "
        "If the Researcher reported 'State Deficiency' for the current section's topic, "
        "assign a different section where RAG evidence exists. "
        "If all topics from the progress summary have been written or have State Deficiency, "
        "output only `[DRAFTING_COMPLETE]` and nothing else."
    ),
    (PhaseType.DRAFTING, "Data Analyst"): (
        "Synthesize the Researcher's evidence from '### Received messages' into a dense "
        "structured Markdown list of claims, data points, and citations, scoped strictly "
        "to the current LeadArchitect directive. "
        "CRITICAL: Do NOT generate specific numerical values (percentages, µs latencies, "
        "ratios, correlation coefficients, thresholds) unless they appear verbatim in "
        "the Researcher's message. If a metric is absent, write "
        "'State Deficiency: [metric name]' — never substitute a plausible-sounding number. "
        "CRITICAL: If the current team objective is a placeholder rather than a real "
        "section title (e.g., 'NEXT_SECTION_ASSIGNMENT'), output only "
        "`[WAITING_FOR_DIRECTIVE]` and nothing else. "
        "CRITICAL: If the Researcher's message contains no evidence "
        "(only State Deficiency entries), output only `[NO NEW EVIDENCE]` and nothing else."
    ),
    (PhaseType.DRAFTING, "Researcher"): (
        "If DataAnalyst flags a gap, retrieve the missing evidence from RAG and "
        "forward it. Otherwise, hold — do not add noise."
    ),
    (PhaseType.DRAFTING, "Collector"): (
        "Write the current report section as polished scientific prose using the "
        "DataAnalyst's structured input. Append it to the report. "
        "ALWAYS begin the section with a Markdown heading that matches the section "
        "title assigned by the LeadArchitect (e.g., `## Section Title`). "
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
        "If the section is factually correct, logically coherent, and scientifically "
        "rigorous, output ONLY `[NO_REVISION_NEEDED]` and nothing else. "
        "Otherwise, for each issue state: (1) the specific problem, "
        "(2) the exact correction required. Be concise and actionable."
    ),
    # SECTION_REVIEW — revision round
    (PhaseType.SECTION_REVIEW, "Data Analyst"): (
        "Prepare corrected content for the section currently under review. "
        "Output a structured Markdown list containing ONLY claims directly "
        "supported by RAG evidence — omit anything the Reviewer flagged as unverifiable. "
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
        "If DataAnalyst's output is `[NO SUPPORTED CONTENT]`, output nothing — "
        "leave the section unchanged rather than adding meta-commentary. "
        "Do NOT write any sentence that references 'the next section', 'the following "
        "section', 'the subsequent section', or previews future content. "
        "CRITICAL: Do NOT write about the retrieval process, the RAG system, the absence "
        "of data, or any pipeline-internal details."
    ),
    # VALIDATION
    (PhaseType.VALIDATION, "Reviewer"): (
        "Read the full report and identify ONLY cross-section quality issues: "
        "abrupt transitions, duplicated content between sections, broken narrative flow, "
        "or contradictions between sections. "
        "Do NOT repeat section-level factual checks — those were already done. "
        "For each issue, state the affected sections and the specific problem. "
        "If the report reads well globally, state so briefly."
    ),
    (PhaseType.VALIDATION, "Lead Architect"): (
        "Receive the Reviewer's global quality notes and write a brief validation "
        "conclusion (2-4 sentences). Summarise the overall quality of the report "
        "and list any critical remaining concerns. This is a diagnostic output only — "
        "do not trigger further rewrites."
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
        base = ROLE_DESCRIPTION.get(role, "")
        phase = self._current_phase()
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

        if phase == PhaseType.VALIDATION:
            from utils.globals import ReportState
            block += f"\n**Report sections:**\n"
            block += ReportState.instance().list_sections(verbose=True) + "\n"

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
