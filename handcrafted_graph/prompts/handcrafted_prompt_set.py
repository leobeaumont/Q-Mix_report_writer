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
    PhaseType.REVIEW: (
        "### Pipeline Phase: REVIEW\n"
        "The team is auditing the complete draft. "
        "Focus on factual accuracy, logical coherence, and scientific rigour. "
        "Generate specific, actionable feedback — do not rewrite content directly."
    ),
    PhaseType.REVISION: (
        "### Pipeline Phase: REVISION\n"
        "The team is applying corrections identified during review. "
        "Target only the flagged issues. Avoid introducing new content outside the scope "
        "of the reviewer's feedback."
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
        "End by setting the team's first section target."
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
        "Return raw evidence atoms with source attribution. "
        "Signal 'State Deficiency' if the knowledge base lacks the requested data."
    ),
    (PhaseType.RESEARCH, "Data Analyst"): (
        "Synthesise the Researcher's evidence atoms into a structured Markdown list "
        "of logical claims, data points, and gaps. This becomes the writing blueprint."
    ),
    # DRAFTING
    (PhaseType.DRAFTING, "Lead Architect"): (
        "Check the 'Sections written so far' list in your context. "
        "Assign the NEXT section that has NOT yet been written — never repeat a topic "
        "already present in the list. "
        "State the new section title and provide a one-sentence structural directive "
        "for DataAnalyst. Do not write prose yourself. "
        "Do NOT invent document names, section numbers, page numbers, equation numbers, "
        "or formulas that have not yet appeared in RAG results — page references are "
        "the Researcher's responsibility, not yours. "
        "Do NOT invent names of external frameworks, papers, or authors — including as "
        "illustrative examples in directives. If a gap exists, document the limitation "
        "using only terms that appeared in RAG results. "
        "If the Researcher reported 'State Deficiency' for the current section's topic, "
        "assign a different section where RAG evidence exists rather than continuing "
        "with the unsupported one."
    ),
    (PhaseType.DRAFTING, "Data Analyst"): (
        "Produce a dense, structured Markdown list of claims, data points, and "
        "citations the Collector should include in THIS section only. "
        "Check 'Sections written so far' — do not reproduce topics or claims already "
        "covered in a completed section. Only output NEW material scoped strictly "
        "to the current LeadArchitect directive. "
        "CRITICAL: Do NOT generate specific numerical values (percentages, µs latencies, "
        "ratios, correlation coefficients, thresholds) unless they appear verbatim in "
        "the Researcher's current message. If a metric is absent from the Researcher's "
        "output, write 'State Deficiency: [metric name]' — never substitute a "
        "plausible-sounding number. Your own previous output is an unverified draft; "
        "do not treat numbers from it as sourced facts."
    ),
    (PhaseType.DRAFTING, "Researcher"): (
        "If DataAnalyst flags a gap, retrieve the missing evidence from RAG and "
        "forward it. Otherwise, hold — do not add noise."
    ),
    (PhaseType.DRAFTING, "Collector"): (
        "Write the current report section as polished scientific prose using the "
        "DataAnalyst's structured input. Append it to the report. "
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
    # REVIEW
    (PhaseType.REVIEW, "Reviewer"): (
        "Perform a full audit of the report draft. For each issue, state: "
        "(1) the location, (2) the specific problem, (3) the correction required. "
        "Score each section on logic, verifiability, and precision."
    ),
    (PhaseType.REVIEW, "Lead Architect"): (
        "Receive the reviewer's critique and decide which issues require revision. "
        "Set the team objective to the highest-priority correction."
    ),
    # REVISION
    (PhaseType.REVISION, "Lead Architect"): (
        "Direct DataAnalyst to prepare corrected content for the specific section "
        "flagged by the Reviewer. Scope the correction narrowly. "
        "Reference the exact section ID from the report section list above "
        "(e.g., 'correct section_2')."
    ),
    (PhaseType.REVISION, "Data Analyst"): (
        "Prepare corrected content addressing the reviewer's stated concern. "
        "Provide a structured Markdown list ready for Collector to rewrite. "
        "REQUIRED: Begin your output with the tag `[SECTION_ID: section_X]` "
        "(using the exact ID from the section list above) to specify which section "
        "is being corrected. "
        "For each claim the Reviewer flagged as unverified or absent from RAG: "
        "emit `REMOVE: [exact claim text]` — do NOT relabel it as speculative or "
        "paraphrase it. Only include claims that are directly supported by RAG evidence "
        "in this run's Researcher output."
    ),
    (PhaseType.REVISION, "Researcher"): (
        "Retrieve additional evidence from RAG only if the revision requires new "
        "sourcing. Otherwise skip."
    ),
    (PhaseType.REVISION, "Collector"): (
        "Rewrite the flagged section using the corrected content from DataAnalyst. "
        "Parse the `[SECTION_ID: section_X]` tag at the top of DataAnalyst's message "
        "to identify which section to replace. "
        "The rewritten section will replace the original in-place — do not append. "
        "CRITICAL: Any line in DataAnalyst's output that begins with `REMOVE:` marks a "
        "claim that must be deleted from the rewritten section — do not include it, "
        "do not paraphrase it, do not add a caveat around it. Excise it entirely. "
        "Do NOT write any sentence that references 'the next section', 'the following "
        "section', 'the subsequent section', or previews future content. "
        "CRITICAL: Do NOT write about the retrieval process, the RAG system, the absence "
        "of data, or any pipeline-internal details. If the corrected content from "
        "DataAnalyst is insufficient to improve the section, output nothing — "
        "leave the section unchanged rather than filling it with meta-commentary."
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

        # Show section IDs in REVIEW and REVISION so agents can target corrections.
        if phase in (PhaseType.REVIEW, PhaseType.REVISION):
            from utils.globals import ReportState
            block += f"\n**Report sections (use IDs for targeted corrections):**\n"
            block += ReportState.instance().list_sections() + "\n"

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
