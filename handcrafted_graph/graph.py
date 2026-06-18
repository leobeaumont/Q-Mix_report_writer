"""
HandcraftedGraph — phase-based multi-agent communication graph.

Replaces QMIX action selection with a hand-designed topology that is run as a
deterministic, phase-ordered pipeline. The graph reuses the existing Node /
AgentRegistry / Collector infrastructure unchanged; only the edge-wiring and
execution scheduling differ from QMIXGraph.

Usage:
    graph = HandcraftedGraph(
        llm_name="qwen3:8b",
        agent_names=["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"],
    )
    answers, total_tokens = await graph.arun({"task": "Write a report on X"})
"""

from __future__ import annotations

import asyncio
import re
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import shortuuid
from tqdm import tqdm

from graph.node import Node
from handcrafted_graph.phases import PhaseConfig, PhaseType, RoundTopology, PHASE_SEQUENCE
from handcrafted_graph.scheduler import RoundScheduler, SkipStrategy
from handcrafted_graph.state import PhaseState
from handcrafted_graph.prompts.handcrafted_prompt_set import _extract_section_directive
from agents.collector import _ABSENCE_RE
from utils.globals import PromptTokens, CompletionTokens, ReportState, ExecutionTrace, SourceBuffer

logger = logging.getLogger("handcrafted_graph")

# ------------------------------------------------------------------
# Citation tagging — sentence-level n-gram overlap constants
# ------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(\s+)(?=[A-Z"\'])')
# Matches an explicit removal instruction in a Reviewer critique. Used to
# authorize [REMOVE_SECTION] — the Collector refuses the sentinel otherwise.
_REMOVAL_REQUEST_RE = re.compile(
    r'\b(?:remove|delete|drop|removal|deletion)\b[^.\n]{0,80}\bsection\b'
    r'|\bsection\b[^.\n]{0,80}\b(?:removed?|deleted?|dropped|removal|deletion)\b',
    re.IGNORECASE,
)
# A line led by an all-caps bracketed sentinel, e.g. "[NO NEW EVIDENCE]",
# "[WAITING_FOR_DIRECTIVE]", or "[RESEARCH_EXHAUSTED] RAG returned no documents."
# (the trailing prose the Researcher appends). Used to detect a DataAnalyst
# Round-A output that carries no blueprint content. Requires >=3 chars inside the
# brackets so a stray single-letter token like "[A]" is not mistaken for a sentinel.
_BRACKET_SENTINEL_RE = re.compile(r'^\[[A-Z0-9_ ]{3,}\]')
_TOKEN_RE          = re.compile(r'[a-zA-Z0-9]+')
_MIN_TOKEN_LEN        = 5   # minimum token length to be considered meaningful
_MIN_CITATION_OVERLAP = 5   # shared tokens required to assign a citation
_MIN_SENTENCE_TOKENS  = 5   # sentence must have this many tokens to be a candidate

# The model sometimes names a source inline in its prose (e.g. quoting the
# filename and page it drew from) rather than leaving the sentence for the
# citation pass to tag. _apply_citation_tags() rewrites these inline references
# into proper [cite:N, p.X] tags alongside the overlap-based tagging. A source
# filename is an arXiv-style "*.pdf". Two surface forms are handled:
#   1. Bracketed:  "[2105.06979.pdf | Page: 26]", "[2605.30554.pdf, Page 2 and 11]",
#                  "[1711.02644.pdf, Page 4]", "[2605.30554.pdf]",
#                  "[Source: 9308022.pdf | Page: 29]" (optional "Source:" prefix)
#   2. Prose:      "Source 2105.06979.pdf (Page 2)", "(2605.26692.pdf)", "Source 0410066.pdf"
_INLINE_FILE = r'[A-Za-z0-9][A-Za-z0-9._-]*\.pdf'
_INLINE_PAGES = r'(?:\|\s*|,\s*)?(?:Pages?|pp?\.)\s*:?\s*([0-9][0-9,\s]*(?:and\s*[0-9]+)?)'
_INLINE_REF_BRACKET_RE = re.compile(
    rf'\[\s*(?:Sources?\s*:?\s*)?({_INLINE_FILE})\s*(?:{_INLINE_PAGES})?\s*\]',
    re.IGNORECASE,
)
_INLINE_REF_SOURCE_RE = re.compile(
    rf'(?:Sources?\s+)({_INLINE_FILE})\s*(?:\(\s*{_INLINE_PAGES}\s*\))?',
    re.IGNORECASE,
)
_INLINE_REF_PAREN_RE = re.compile(rf'\(\s*({_INLINE_FILE})\s*\)')

# Bare bracketed reference markers the model sometimes copies verbatim from a
# retrieved chunk's own reference list, e.g. "Ref. [32]", "[12, 14]". The number
# indexes that source's bibliography, never our [cite:N] scheme, so any such
# marker whose number is not among the [cite:N] tags applied to the same
# sentence is a stray reference dropped by _strip_orphan_citation_markers().
# The leading "\d" requirement means real "[cite:N]" tags (which start with "c")
# are never matched.
_ORPHAN_CITE_RE = re.compile(
    r'(?:Refs?\.?|References?)?\s*\[\s*\d+(?:\s*[,&]\s*\d+)*\s*\]',
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    # 2–3 letter function words
    "the", "and", "for", "are", "was", "its", "his", "her", "our", "has",
    "had", "not", "but", "all", "can", "may", "one", "two", "any", "few",
    "new", "via", "per", "non", "sub", "pre", "pro", "out", "off", "set",
    "let", "yet", "nor", "use", "due", "far", "low", "top", "end", "key",
    # 4+ letter common words
    "that", "this", "with", "from", "have", "been", "which", "they", "their",
    "also", "both", "such", "more", "when", "where", "than", "then", "into",
    "onto", "upon", "these", "those", "there", "here", "what", "some", "each",
    "over", "after", "under", "about", "through", "between", "along", "while",
    "since", "using", "within", "without", "toward", "above", "below",
    "during", "given", "often", "many", "most", "other", "only", "very",
    "show", "shows", "shown", "note", "noted", "term", "terms", "well",
    "thus", "hence", "which", "where", "when", "how", "now", "then",
})


class NoCorpusCoverageError(RuntimeError):
    """Raised when PLANNING produces no section outline.

    This means the Researcher's coverage scan found no documents in the
    knowledge base relevant to the requested task, so the LeadArchitect could
    not ground an outline. Rather than silently drafting an empty report (and
    later hallucinating a validation directive over non-existent sections), the
    pipeline aborts here with a clear, actionable message.
    """


class HandcraftedGraph:
    """Phase-based multi-agent graph with a hand-crafted communication topology.

    Args:
        llm_name: LLM model name forwarded to every agent node.
        agent_names: Ordered list of agent names to instantiate.  Must include
                     "Collector" exactly once.
        skip_strategy: How optional agents decide whether to participate.
        execution_trace: Whether to record an execution trace for analysis.
        phases: Override the default phase sequence (useful for ablations).
    """

    def __init__(
        self,
        llm_name: str,
        agent_names: List[str],
        skip_strategy: SkipStrategy = SkipStrategy.ALWAYS_INCLUDE,
        execution_trace: bool = False,
        phases: Optional[List[PhaseConfig]] = None,
    ) -> None:
        self.id = shortuuid.ShortUUID().random(length=4)
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.nodes: Dict[str, Node] = {}
        self.collector_id: Optional[str] = None
        self.skip_strategy = skip_strategy
        self.phases = phases or PHASE_SEQUENCE

        self._init_nodes()
        self._inject_prompt_set()

        self.node_ids = list(self.nodes.keys())
        self.phase_state = PhaseState.instance()
        self.execution_trace = ExecutionTrace.instance() if execution_trace else None

        # Per-round LeadArchitect outputs captured during PLANNING, used to recover
        # the section outline even when the final round emits a signal, not an outline.
        self._planning_la_outputs: List[str] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_nodes(self) -> None:
        from agents.agent_registry import AgentRegistry

        for i, agent_name in enumerate(self.agent_names):
            try:
                node_id = f"{agent_name}_{i}"
                node = AgentRegistry.get(agent_name, id=node_id, llm_name=self.llm_name)
                self.nodes[node_id] = node
                if agent_name == "Collector":
                    self.collector_id = node_id
            except Exception as exc:
                logger.warning(f"Failed to create agent '{agent_name}': {exc}")

    def _inject_prompt_set(self) -> None:
        """Replace the default 'redacting' prompt set with the phase-aware one."""
        # Importing the module registers the prompt set as a side-effect.
        import handcrafted_graph.prompts.handcrafted_prompt_set  # noqa: F401
        from prompt.prompt_set_registry import PromptSetRegistry

        prompt_set = PromptSetRegistry.get("handcrafted_redacting")
        for node in self.nodes.values():
            node.prompt_set = prompt_set

    # ------------------------------------------------------------------
    # Main execution entry point
    # ------------------------------------------------------------------

    async def arun(
        self,
        input: Dict[str, str],
        max_tries: int = 3,
        max_time: int = 300,
        max_validation_attempts: int = 1,
    ) -> Tuple[List[Any], int]:
        """Run the full phase pipeline and return the finished report.

        Args:
            input: {"task": "<report subject>"}
            max_tries: Retry attempts per node on failure.
            max_time: Per-node execution timeout in seconds.

        Returns:
            (answers, total_tokens) where answers is a single-element list
            containing the final report text.
        """
        tokens_before = PromptTokens.instance().value + CompletionTokens.instance().value
        self.phase_state.reset()

        writing_phases    = [p for p in self.phases if not p.section_aware
                             and p.name != PhaseType.VALIDATION]
        correction_phases = [p for p in self.phases if p.section_aware
                             or p.name == PhaseType.VALIDATION]

        # ── Bar 1: Writing (PLANNING → RESEARCH → DRAFTING) ───────────────
        # Total is fully known upfront — no patching needed mid-run.
        writing_total = sum(p.max_rounds for p in writing_phases)
        writing_pbar = tqdm(
            total=writing_total, desc="Writing report", unit="round", leave=True
        )
        try:
            for phase in writing_phases:
                if phase.name == PhaseType.DRAFTING:
                    # Flush sources accumulated during PLANNING and RESEARCH into the
                    # global bibliography before any section is written. These documents
                    # informed the outline and evidence synthesis and may be cited in the
                    # report even if not re-retrieved during a specific drafting round.
                    pre_draft_sources = SourceBuffer.instance().flush()
                    if pre_draft_sources:
                        ReportState.instance().sources += pre_draft_sources
                        logger.info(
                            f"[{self.id}] Flushed {len(pre_draft_sources)} pre-draft source(s) "
                            f"into global bibliography."
                        )
                self.phase_state.set_phase(phase.name)
                logger.info(f"[{self.id}] Starting phase: {phase.name.value.upper()}")
                await self._execute_phase(input, phase, max_tries, max_time, writing_pbar)

                # After PLANNING: parse the section outline from the LA's last response
                # and store it in ReportState before RESEARCH buries the context.
                # Also patch writing_pbar.total so the bar reflects actual DRAFTING rounds.
                if phase.name == PhaseType.PLANNING:
                    # The outline is normally produced in the latest PLANNING round,
                    # but a later round may instead emit a directive or signal (e.g.
                    # [AWAITING_COVERAGE_DATA]) with no outline. node.outputs is reset
                    # every round, so last_memory only ever holds the final round —
                    # we rely on the per-round LA outputs captured during the phase.
                    # Prefer the most recent round whose output parses into a section
                    # list; fall back to earlier rounds when the latest has none.
                    planned: List[str] = []
                    for out in reversed(self._planning_la_outputs):
                        candidate = self._parse_section_titles(str(out or ""))
                        if candidate:
                            planned = candidate
                            break
                    ReportState.instance().planned_sections = planned
                    logger.info(
                        f"[{self.id}] Extracted {len(planned)} planned section(s) from PLANNING."
                    )
                    # Fail fast on empty corpus coverage. An empty outline means the
                    # Researcher's coverage scan confirmed no relevant documents (the
                    # LeadArchitect emits [AWAITING_COVERAGE_DATA] instead of a list).
                    # Continuing would draft an empty report and then hallucinate a
                    # validation directive over sections that do not exist — abort with
                    # a clear message instead.
                    if not planned:
                        task_desc = str(input.get("task", "")).strip()[:120]
                        raise NoCorpusCoverageError(
                            f"PLANNING produced no section outline for task "
                            f"{task_desc!r}. The knowledge base appears to contain no "
                            f"documents relevant to this subject, so no grounded report "
                            f"can be written. Aborting before drafting an empty report."
                        )
                    drafting_phase = next(
                        (p for p in writing_phases if p.name == PhaseType.DRAFTING), None
                    )
                    if drafting_phase:
                        delta = 2 * len(planned) - drafting_phase.max_rounds
                        writing_pbar.total += delta
                        writing_pbar.refresh()
        finally:
            writing_pbar.close()

        # ── Transition notice ─────────────────────────────────────────────
        n_sections = len(ReportState.instance().sections)
        tqdm.write(
            f"\n  Writing complete — {n_sections} section(s) drafted."
            f"  Starting review & correction phase…\n"
        )

        report_state = ReportState.instance()

        # Split correction phases so the validation loop can restart only SECTION_REVIEW.
        section_review_phases = [p for p in correction_phases if p.section_aware]
        validation_phases     = [p for p in correction_phases if p.name == PhaseType.VALIDATION]

        # ── Validation loop ────────────────────────────────────────────────
        for attempt in range(max_validation_attempts + 1):
            sections   = report_state.sections
            n_sections = len(sections)

            # Reset per-pass validation state so each pass is a fresh assessment.
            report_state.validation_notes  = []
            report_state.validation_window = None

            # Trace: insert a pass-boundary marker before every re-review so the
            # visualizer can distinguish first-pass rounds from retry rounds.
            if self.execution_trace is not None and attempt > 0:
                self.execution_trace.trace.append({
                    "exec_order": ["__validation_retry__"],
                    "validation_pass": attempt + 1,
                    "validation_directive": report_state.validation_directive,
                })

            # Build progress bar for this pass.
            sr_total = sum(p.max_rounds * n_sections for p in section_review_phases)
            val_total = 0
            for p in validation_phases:
                if p.window_aware:
                    windows = self._validation_windows(report_state, p)
                    val_total += len(windows) + 1
                else:
                    val_total += p.max_rounds

            pass_label = (
                "Review & correction"
                if attempt == 0
                else f"Re-review (pass {attempt + 1}/{max_validation_attempts + 1})"
            )
            correction_pbar = tqdm(
                total=sr_total + val_total, desc=pass_label, unit="round", leave=True
            )
            try:
                for phase in section_review_phases:
                    self.phase_state.set_phase(phase.name)
                    logger.info(f"[{self.id}] Starting phase: {phase.name.value.upper()}")
                    await self._execute_phase(input, phase, max_tries, max_time, correction_pbar)

                for phase in validation_phases:
                    self.phase_state.set_phase(phase.name)
                    logger.info(f"[{self.id}] Starting phase: {phase.name.value.upper()}")
                    await self._execute_phase(input, phase, max_tries, max_time, correction_pbar)
            finally:
                correction_pbar.close()

            # ── Check validation outcome ───────────────────────────────────
            la_node = self._get_node_by_name("LeadArchitect")
            la_output = (
                str(la_node.outputs[-1] or "")
                if la_node and la_node.outputs else ""
            )

            # Trace: stamp the outcome onto the last round so the visualizer can
            # colour-code pass vs fail without parsing the LeadArchitect response.
            if self.execution_trace is not None:
                outcome = (
                    "PASSED" if "[VALIDATION_PASSED]" in la_output
                    else "MAX_ATTEMPTS" if attempt >= max_validation_attempts
                    else "FAILED"
                )
                self.execution_trace.trace[-1]["validation_outcome"] = outcome
                self.execution_trace.trace[-1]["validation_attempt"] = attempt + 1

            if "[VALIDATION_PASSED]" in la_output:
                logger.info(f"[{self.id}] Validation passed on attempt {attempt + 1}.")
                tqdm.write(f"\n  Validation passed — report finalised.\n")
                break

            if attempt >= max_validation_attempts:
                tqdm.write(
                    f"\n  Max validation attempts ({max_validation_attempts + 1}) reached"
                    f" — finalising report as-is.\n"
                )
                break

            # ── Validation failed — decompose and prepare next pass ────────
            rv_node = self._get_node_by_name("Reviewer")
            reviewer_synthesis = (
                str(rv_node.outputs[-1] or "")
                if rv_node and rv_node.outputs else ""
            )
            combined_issues = f"{reviewer_synthesis}\n\n{la_output}".strip()
            report_state.validation_issues = combined_issues  # used to scope the next validation pass

            tqdm.write(
                f"\n  Validation failed (attempt {attempt + 1}/{max_validation_attempts + 1})."
                f"  Decomposing issues and scheduling re-review…\n"
            )

            directive = await self._decompose_validation_directive(combined_issues, report_state)
            report_state.validation_directive = directive
            tqdm.write(f"  Revision directive:\n{directive}\n")

            # Trace: dedicated entry for the decomposition call so the visualizer
            # can show the directive and the issues that triggered it.
            if self.execution_trace is not None:
                self.execution_trace.trace.append({
                    "exec_order": ["__decomposition__"],
                    "decomposition": {
                        "attempt": attempt + 1,
                        "issues_summary": combined_issues[:600],
                        "directive": directive,
                    },
                })

            self._clear_all_memory()

        # ── Assemble final report ──────────────────────────────────────────
        self._build_bibliography()
        report = report_state.content

        # Reader-facing abstract: generated over the FINAL body (post-validation,
        # post-revision) so it reflects the report as actually shipped. Prepended
        # as the first `## Abstract` section; the PDF exporter renders the title
        # page separately, so this lands at the top of the body.
        abstract = await self._generate_abstract(report_state)
        if abstract:
            report = f"## Abstract\n\n{abstract}\n\n" + report.lstrip()

        if report_state.bibliography:
            report = report.rstrip() + "\n\n" + report_state.bibliography

        # Stamp the final report onto the last trace round's Collector slot. The
        # last entry is not always a normal round: pass-boundary markers
        # (__decomposition__, __validation_retry__) carry no "Collector" key, and
        # when the report has 0 sections the correction phases skip without
        # appending a round — leaving a marker as trace[-1]. Guard accordingly so
        # finalisation never raises KeyError.
        if self.execution_trace is not None and self.execution_trace.trace:
            last_round = self.execution_trace.trace[-1]
            if "Collector" in last_round:
                last_round["Collector"]["report_state"] = report

        tokens_after = PromptTokens.instance().value + CompletionTokens.instance().value
        total_tokens = int(tokens_after - tokens_before)

        return ([report] if report else ["No report generated"]), total_tokens

    # ------------------------------------------------------------------
    # Phase and round execution
    # ------------------------------------------------------------------

    async def _execute_phase(
        self,
        input: Dict[str, str],
        phase: PhaseConfig,
        max_tries: int,
        max_time: int,
        overall_pbar: Optional[tqdm] = None,
    ) -> None:
        self._clear_all_memory()

        scheduler = RoundScheduler(
            nodes=self.nodes,
            collector_id=self.collector_id,
            skip_strategy=self.skip_strategy,
            llm=self._get_any_llm(),
        )

        if phase.name == PhaseType.DRAFTING:
            await self._execute_drafting_phase(
                input, phase, scheduler, max_tries, max_time, overall_pbar
            )
            return

        if phase.section_aware:
            await self._execute_section_aware_phase(
                input, phase, scheduler, max_tries, max_time, overall_pbar
            )
            return

        if phase.window_aware:
            await self._execute_window_aware_phase(
                input, phase, scheduler, max_tries, max_time, overall_pbar
            )
            return

        n_patterns = len(phase.round_topologies)

        # Capture each PLANNING round's LeadArchitect output so the outline can be
        # recovered later even if the final round emitted a signal instead of the
        # outline. node.outputs is wiped at the start of every round, so this is
        # the only place the per-round outline survives.
        if phase.name == PhaseType.PLANNING:
            self._planning_la_outputs = []

        for round_idx in range(phase.max_rounds):
            topology = phase.round_topologies[round_idx % n_patterns]

            active_agents = await scheduler.get_active_agents(
                topology, round_idx, task_input=input
            )

            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[{phase.name.value.upper()}] round {round_idx + 1}/{phase.max_rounds}"
                )

            if not active_agents:
                logger.info(
                    f"Phase '{phase.name.value}' ended early at round {round_idx} "
                    f"(no active agents)."
                )
                if overall_pbar is not None:
                    overall_pbar.update(1)
                break

            logger.info(
                f"  [{phase.name.value}] Round {round_idx}: "
                f"active={sorted(active_agents)}"
            )

            self._build_topology(topology, active_agents)
            self._connect_temporal()

            # Initialise trace slot and pre-populate spatial edges before nodes run.
            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()

            await self._execute_round(input, active_agents, max_tries, max_time)

            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()

            if phase.name == PhaseType.PLANNING:
                la_node = self._get_node_by_name("LeadArchitect")
                la_out = (
                    str(la_node.outputs[-1] or "")
                    if la_node and la_node.outputs else ""
                )
                self._planning_la_outputs.append(la_out)

            if overall_pbar is not None:
                overall_pbar.update(1)

            if phase.name == PhaseType.RESEARCH:
                researcher_node = self._get_node_by_name("Researcher")
                if researcher_node and researcher_node.outputs:
                    latest_output = str(researcher_node.outputs[-1] or "")
                    if "[RESEARCH_EXHAUSTED]" in latest_output:
                        logger.info(
                            f"Phase '{phase.name.value}' completed early at round {round_idx}: "
                            f"[RESEARCH_EXHAUSTED] signal received."
                        )
                        if overall_pbar is not None:
                            overall_pbar.update(phase.max_rounds - round_idx - 1)
                        break

    # ------------------------------------------------------------------
    # Drafting phase execution (code-driven section iteration)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_section_titles(text: str) -> List[str]:
        """Extract ordered section titles from a numbered outline.

        Tries bold format first (1. **Title**), then plain numbered list.
        """
        titles = re.findall(r'^\d+[.)]\s+\*\*(.+?)\*\*', text, re.MULTILINE)
        if titles:
            return [t.strip() for t in titles]
        titles = re.findall(r'^\d+[.)]\s+(.+?)$', text, re.MULTILINE)
        return [t.strip().rstrip('.,;:') for t in titles if t.strip()]

    @staticmethod
    def _drafting_blueprint_is_usable(da_output: str) -> bool:
        """Return True when the DataAnalyst's DRAFTING Round-A output is a usable blueprint.

        Mirrors Collector._data_analyst_has_content: the output is usable when at
        least one non-empty line is neither a bare sentinel (e.g. [NO NEW EVIDENCE],
        [WAITING_FOR_DIRECTIVE], an echoed [RESEARCH_EXHAUSTED]) nor an absence /
        State-Deficiency marker. A blueprint that mixes real claims with a few
        'State Deficiency: ...' lines still counts.

        When this returns True the write round reuses the existing blueprint and
        skips the redundant Researcher + DataAnalyst re-query. When it returns
        False (pure-sentinel or empty output) the write round re-runs research as
        a genuine second retrieval attempt for that section.
        """
        text = (da_output or "").strip()
        if not text:
            return False
        for line in text.splitlines():
            line = line.strip().strip("-").strip()
            if not line:
                continue
            if _BRACKET_SENTINEL_RE.match(line):
                continue          # bare sentinel line, e.g. [NO NEW EVIDENCE]
            if _ABSENCE_RE.search(line):
                continue          # absence / State Deficiency line
            return True           # a real content line — blueprint is usable
        return False

    async def _execute_drafting_phase(
        self,
        input: Dict[str, str],
        phase: PhaseConfig,
        scheduler: RoundScheduler,
        max_tries: int,
        max_time: int,
        overall_pbar: Optional[tqdm] = None,
    ) -> None:
        """Execute DRAFTING by iterating ReportState.planned_sections in order.

        For each planned section:
          - Sets report_state.task to the section title so all agents know the target.
          - Round A (index 0): LeadArchitect formulates the task; Researcher + DataAnalyst
            prepare evidence.
          - Round B (index 1): DataAnalyst + Collector write the section.
        Memory is cleared between sections to prevent cross-section contamination.
        Exits naturally when all planned sections have been attempted.
        Falls back to the generic round loop if no planned sections were extracted.
        """
        report_state = ReportState.instance()
        planned = report_state.planned_sections

        if not planned:
            logger.warning(
                f"[{self.id}] DRAFTING: no planned sections found — "
                f"falling back to generic round loop."
            )
            n_patterns = len(phase.round_topologies)
            for round_idx in range(phase.max_rounds):
                topology = phase.round_topologies[round_idx % n_patterns]
                # Refresh the bar label: without this the bar keeps the previous
                # phase's description (e.g. "[RESEARCH] round 6/6") for the whole
                # generic-fallback draft instead of showing DRAFTING progress.
                if overall_pbar is not None:
                    overall_pbar.set_description(
                        f"[{phase.name.value.upper()}] round {round_idx + 1}/{phase.max_rounds}"
                    )
                active_agents = await scheduler.get_active_agents(
                    topology, round_idx, task_input=input
                )
                if not active_agents:
                    break
                self._build_topology(topology, active_agents)
                self._connect_temporal()
                if self.execution_trace is not None:
                    self._init_trace_round()
                    self._trace_spatial_edges()
                await self._execute_round(input, active_agents, max_tries, max_time)
                self._update_memory()
                self._clear_spatial()
                self.phase_state.increment_round()
                if overall_pbar is not None:
                    overall_pbar.update(1)
            return

        n_sections = len(planned)
        prep_topology  = phase.round_topologies[0]   # Round A: LA + Researcher + DA
        write_topology = phase.round_topologies[1]   # Round B: DA + Collector

        for i, section_title in enumerate(planned):
            report_state.drafting_section_idx = i
            report_state.task = f"[NEXT SECTION TO WRITE: {section_title}]"
            self._clear_all_memory()

            # ---- Round A: prep -------------------------------------------
            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[DRAFTING] section {i + 1}/{n_sections} — prep"
                )
            active_agents = await scheduler.get_active_agents(
                prep_topology, 0, task_input=input
            )
            logger.info(
                f"  [drafting] Section {i + 1}/{n_sections} prep: "
                f"active={sorted(active_agents)}"
            )
            self._build_topology(prep_topology, active_agents)
            self._connect_temporal()
            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()
            await self._execute_round(input, active_agents, max_tries, max_time)
            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()
            if overall_pbar is not None:
                overall_pbar.update(1)

            # ---- Round B: write ------------------------------------------
            # If Round A produced a usable blueprint, skip the redundant Round B
            # Researcher + DataAnalyst re-query and forward that blueprint straight
            # to the Collector. The Round B research pass is reserved for the case
            # where Round A failed to produce a blueprint (e.g. the Researcher hit
            # [RESEARCH_EXHAUSTED] and the DataAnalyst emitted [NO NEW EVIDENCE]),
            # giving that section a genuine second retrieval attempt.
            da_node = self._get_node_by_name("DataAnalyst")
            da_output = (
                str(da_node.outputs[-1] or "")
                if da_node is not None and da_node.outputs else ""
            )
            blueprint_ready = self._drafting_blueprint_is_usable(da_output)

            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[DRAFTING] section {i + 1}/{n_sections} — write"
                )

            if blueprint_ready:
                # Reuse Round A's blueprint. The DataAnalyst node still holds that
                # output in node.outputs (memory is not cleared between a section's
                # two rounds), so wiring a one-way DataAnalyst → Collector spatial
                # edge delivers it via get_spatial_info() exactly as if the
                # DataAnalyst had produced it this round — without re-executing it.
                # This is the same forwarding pattern used in SECTION_REVIEW
                # revision rounds (Reviewer → DataAnalyst).
                active_agents = {"Collector"}
                logger.info(
                    f"  [drafting] Section {i + 1}/{n_sections} write: "
                    f"reusing Round A blueprint — skipping Researcher/DataAnalyst."
                )
                self._build_topology(write_topology, active_agents)
                collector_node = self.nodes.get(self.collector_id)
                if da_node is not None and collector_node is not None:
                    da_node.add_successor(collector_node, "spatial")
            else:
                active_agents = await scheduler.get_active_agents(
                    write_topology, 1, task_input=input
                )
                logger.info(
                    f"  [drafting] Section {i + 1}/{n_sections} write: "
                    f"Round A produced no blueprint — retrying research. "
                    f"active={sorted(active_agents)}"
                )
                self._build_topology(write_topology, active_agents)

            self._connect_temporal()
            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()
            await self._execute_round(input, active_agents, max_tries, max_time)
            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()
            if overall_pbar is not None:
                overall_pbar.update(1)

        logger.info(
            f"[{self.id}] DRAFTING complete: {n_sections} section(s) attempted, "
            f"{len(report_state.sections)} written."
        )

    async def _execute_section_aware_phase(
        self,
        input: Dict[str, str],
        phase: PhaseConfig,
        scheduler: RoundScheduler,
        max_tries: int,
        max_time: int,
        overall_pbar: Optional[tqdm] = None,
    ) -> None:
        """Execute a section-aware phase by iterating ReportState.sections in order.

        For each section:
          - Round A (index 0): review round — Reviewer audits the section text.
          - Round B (index 1): revision round — skipped when Reviewer signals
            [NO_REVISION_NEEDED]; otherwise DataAnalyst + Collector apply
            corrections in-place via replace_section().

        Memory is cleared between sections to prevent cross-section contamination.
        Within a section's two-round cycle it is preserved so the Reviewer's
        temporal self-edge carries the critique into the revision round.
        """
        report_state = ReportState.instance()
        # Iterate over a snapshot: a [REMOVE_SECTION] revision mutates the live
        # sections list mid-loop, which would shift positions and silently skip
        # the section that follows the removed one. The positional index is
        # re-resolved by ID at every iteration for the same reason.
        sections_snapshot = list(report_state.sections)
        n_sections = len(sections_snapshot)

        if n_sections == 0:
            logger.warning(f"Phase '{phase.name.value}': no sections to review — skipping.")
            return

        review_topology = phase.round_topologies[0]
        revision_topology = phase.round_topologies[1]

        for i, section in enumerate(sections_snapshot):
            idx = next(
                (j for j, s in enumerate(report_state.sections) if s["id"] == section["id"]),
                None,
            )
            if idx is None:
                logger.warning(
                    f"  [{phase.name.value}] Section {i + 1}/{n_sections} ({section['id']}): "
                    f"no longer in the report — skipping."
                )
                if overall_pbar is not None:
                    overall_pbar.update(2)
                continue
            report_state.removal_authorized = False
            directive = report_state.validation_directive
            if directive:
                section_instruction = _extract_section_directive(directive, section["id"])
                if not section_instruction:
                    # No directive for this section — skip both rounds in code.
                    # Relying on the Reviewer to self-police via a prompt instruction
                    # is unreliable — it fact-checks anyway and wastes LLM calls.
                    logger.info(
                        f"  [{phase.name.value}] Section {i + 1}/{n_sections} ({section['id']}): "
                        f"no revision directive — skipping."
                    )
                    if overall_pbar is not None:
                        overall_pbar.update(2)
                    continue

                # Directive exists for this section — bypass Reviewer and DataAnalyst.
                # Give the directive directly to the Collector so the Reviewer cannot
                # override it with its own fact-checking or re-derive the correction.
                report_state.review_section_idx = idx
                self._clear_all_memory()

                if overall_pbar is not None:
                    overall_pbar.set_description(
                        f"[{phase.name.value.upper()}] section {i + 1}/{n_sections} — directive apply"
                    )
                    overall_pbar.update(1)  # account for skipped review round

                # _execute_round filters by node.agent_name, so active_agents must
                # contain the role name string — NOT the node UUID.
                directive_agents = {"Collector"}
                logger.info(
                    f"  [{phase.name.value}] Section {i + 1}/{n_sections} directive apply: "
                    f"active={sorted(directive_agents)}"
                )

                self._build_topology(revision_topology, directive_agents)
                self._connect_temporal()

                if self.execution_trace is not None:
                    self._init_trace_round()
                    self._trace_spatial_edges()

                await self._execute_round(input, directive_agents, max_tries, max_time)
                self._update_memory()
                self._clear_spatial()
                self.phase_state.increment_round()

                if overall_pbar is not None:
                    overall_pbar.update(1)

                self._apply_citation_tags(idx)
                if self.execution_trace is not None:
                    self.execution_trace.trace[-1]["Collector"]["report_state"] = (
                        ReportState.instance().content
                    )
                continue

            report_state.review_section_idx = idx
            self._clear_all_memory()  # clean slate for this section

            # ---- Round A: review ----------------------------------------
            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[{phase.name.value.upper()}] section {i + 1}/{n_sections} — review"
                )

            active_agents = await scheduler.get_active_agents(
                review_topology, 0, task_input=input
            )
            logger.info(
                f"  [{phase.name.value}] Section {i + 1}/{n_sections} review: "
                f"active={sorted(active_agents)}"
            )

            self._build_topology(review_topology, active_agents)
            self._connect_temporal()

            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()

            await self._execute_round(input, active_agents, max_tries, max_time)
            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()

            if overall_pbar is not None:
                overall_pbar.update(1)

            # ---- Check if revision is needed ----------------------------
            reviewer_node = self._get_node_by_name("Reviewer")
            reviewer_output = (
                str(reviewer_node.outputs[-1] or "")
                if reviewer_node and reviewer_node.outputs
                else ""
            )
            # Robust signal check: skip revision only when the response either
            # contains [NO_REVISION_NEEDED] as its sole content, or when any
            # additional content is purely positive confirmation (not corrections).
            # Handles the case where the model appends the signal after real corrections.
            _signal = "[NO_REVISION_NEEDED]"
            _CORRECTION_RE = re.compile(
                r'\*{0,2}Correction:|incorrect|contradicts|hallucinated|misrepresent'
                r'|not verifiable|not supported|not directly supported|overreaches'
                r'|remove the claim|replace.*with',
                re.IGNORECASE,
            )
            # Numbered items like "(1) ..." are the Reviewer's standard format for
            # corrections. If any appear alongside [NO_REVISION_NEEDED], the signal
            # is contradictory — treat the numbered corrections as authoritative.
            _NUMBERED_ITEM_RE = re.compile(r'^\s*\(\d+\)', re.MULTILINE)
            _without_signal = reviewer_output.replace(_signal, "").strip().strip("-").strip()
            _is_clean_pass = (
                _signal in reviewer_output
                and not _CORRECTION_RE.search(_without_signal)
                and not _NUMBERED_ITEM_RE.search(_without_signal)
            )
            if _is_clean_pass:
                logger.info(
                    f"  [{phase.name.value}] Section {i + 1}/{n_sections}: "
                    f"no revision needed — skipping."
                )
                if overall_pbar is not None:
                    overall_pbar.update(1)  # account for skipped revision round
                self._apply_citation_tags(idx)
                if self.execution_trace is not None:
                    self.execution_trace.trace[-1]["Collector"]["report_state"] = (
                        ReportState.instance().content
                    )
                continue

            # ---- Round B: revision --------------------------------------
            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[{phase.name.value.upper()}] section {i + 1}/{n_sections} — revision"
                )

            # Authorize destructive removal only when the critique explicitly
            # asks for it — the Collector refuses [REMOVE_SECTION] otherwise.
            report_state.removal_authorized = bool(
                _REMOVAL_REQUEST_RE.search(reviewer_output)
            )

            active_agents = await scheduler.get_active_agents(
                revision_topology, 1, task_input=input
            )
            logger.info(
                f"  [{phase.name.value}] Section {i + 1}/{n_sections} revision: "
                f"active={sorted(active_agents)}"
            )

            self._build_topology(revision_topology, active_agents)
            # Forward the stored critique from the review round: wire a one-way
            # edge from the (inactive) Reviewer so get_spatial_info() delivers
            # its existing output to DataAnalyst without re-executing the node.
            # Re-running the Reviewer here would cost a full re-review and could
            # produce a different critique than the one this revision is based on.
            da_node = self._get_node_by_name("DataAnalyst")
            if reviewer_node is not None and da_node is not None and reviewer_node.outputs:
                reviewer_node.add_successor(da_node, "spatial")
            self._connect_temporal()

            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()

            await self._execute_round(input, active_agents, max_tries, max_time)
            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()
            report_state.removal_authorized = False

            if overall_pbar is not None:
                overall_pbar.update(1)

            self._apply_citation_tags(idx)
            if self.execution_trace is not None:
                self.execution_trace.trace[-1]["Collector"]["report_state"] = (
                    ReportState.instance().content
                )

    # ------------------------------------------------------------------
    # Window-aware phase execution (VALIDATION)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section_windows(
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

    @staticmethod
    def _revalidation_sections(report_state) -> List[dict]:
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

    def _validation_windows(self, report_state, phase) -> List[List[dict]]:
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
            rv = self._revalidation_sections(report_state)
            return [rv] if rv else []
        return self._build_section_windows(
            report_state.sections, phase.window_size, phase.window_overlap_sections
        )

    async def _execute_window_aware_phase(
        self,
        input: Dict[str, str],
        phase: PhaseConfig,
        scheduler: RoundScheduler,
        max_tries: int,
        max_time: int,
        overall_pbar: Optional[tqdm] = None,
    ) -> None:
        """Execute a window-aware phase by sliding over ReportState.sections.

        For each window (a group of complete sections ≤ window_size chars):
          - Reviewer audits the window content and notes cross-section issues.
          - Notes are accumulated in ReportState.validation_notes.

        After all windows a single synthesis round runs: Reviewer + LeadArchitect
        receive the accumulated notes and produce a consolidated quality report.
        ReportState.validation_window is set to None to signal synthesis mode.

        Memory is cleared between windows to prevent cross-window contamination.
        """
        report_state = ReportState.instance()
        windows = self._validation_windows(report_state, phase)
        n_windows = len(windows)

        if n_windows == 0:
            logger.warning(f"Phase '{phase.name.value}': no sections to validate — skipping.")
            return

        review_topology    = phase.round_topologies[0]
        synthesis_topology = phase.round_topologies[1]

        # ── Window review rounds ──────────────────────────────────────────
        for i, window_sections in enumerate(windows):
            report_state.validation_window = (i, n_windows, window_sections)
            self._clear_all_memory()

            if overall_pbar is not None:
                overall_pbar.set_description(
                    f"[{phase.name.value.upper()}] window {i + 1}/{n_windows}"
                )

            active_agents = await scheduler.get_active_agents(
                review_topology, i, task_input=input
            )
            logger.info(
                f"  [{phase.name.value}] Window {i + 1}/{n_windows}: "
                f"sections={[s['id'] for s in window_sections]} "
                f"active={sorted(active_agents)}"
            )

            self._build_topology(review_topology, active_agents)
            self._connect_temporal()

            if self.execution_trace is not None:
                self._init_trace_round()
                self._trace_spatial_edges()

            await self._execute_round(input, active_agents, max_tries, max_time)
            self._update_memory()
            self._clear_spatial()
            self.phase_state.increment_round()

            reviewer_node = self._get_node_by_name("Reviewer")
            if reviewer_node and reviewer_node.outputs:
                note = str(reviewer_node.outputs[-1] or "").strip()
                if note:
                    report_state.validation_notes.append(note)

            if overall_pbar is not None:
                overall_pbar.update(1)

        # ── Synthesis round ───────────────────────────────────────────────
        report_state.validation_window = None  # signals synthesis mode to agents
        self._clear_all_memory()

        if overall_pbar is not None:
            overall_pbar.set_description(f"[{phase.name.value.upper()}] synthesis")

        active_agents = await scheduler.get_active_agents(
            synthesis_topology, n_windows, task_input=input
        )
        logger.info(
            f"  [{phase.name.value}] Synthesis: active={sorted(active_agents)}"
        )

        self._build_topology(synthesis_topology, active_agents)
        self._connect_temporal()

        if self.execution_trace is not None:
            self._init_trace_round()
            self._trace_spatial_edges()

        await self._execute_round(input, active_agents, max_tries, max_time)
        self._update_memory()
        self._clear_spatial()
        self.phase_state.increment_round()

        if overall_pbar is not None:
            overall_pbar.update(1)

    async def _execute_round(
        self,
        input: Dict[str, str],
        active_agents: Set[str],
        max_tries: int,
        max_time: int,
    ) -> None:
        """Execute all active nodes in topological order (Kahn's algorithm).

        Nodes not in active_agents are completely skipped — their spatial edges
        are simply not built, so they never appear in the execution queue.
        """
        # Build in-degree map over active nodes only.
        active_ids = {
            nid for nid, node in self.nodes.items()
            if node.agent_name in active_agents
        }

        in_degree: Dict[str, int] = {}
        for nid in active_ids:
            node = self.nodes[nid]
            count = 0
            for pred in node.spatial_predecessors:
                if pred.id not in active_ids:
                    continue
                # Ignore self-edge and mutual edge (same tie-breaking as QMIXGraph)
                if (
                    nid in {s.id for s in pred.spatial_successors}
                    and pred.id in {s.id for s in node.spatial_successors}
                    and nid <= pred.id
                ):
                    continue
                count += 1
            in_degree[nid] = count

        queue = [nid for nid in active_ids if in_degree[nid] == 0]
        executed: Set[str] = set()
        pbar = tqdm(total=len(active_ids), desc="Agents", leave=False)

        while queue:
            current_id = queue.pop(0)
            if current_id in executed:
                continue

            t0 = time.time()
            tokens_before = CompletionTokens.instance().value
            for attempt in range(max_tries):
                try:
                    await asyncio.wait_for(
                        self.nodes[current_id].async_execute(
                            input,
                            execution_trace=self.execution_trace,
                        ),
                        timeout=max_time,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        f"Node '{current_id}' attempt {attempt + 1}/{max_tries} "
                        f"failed: {exc}"
                    )

            executed.add(current_id)
            pbar.update()

            if self.execution_trace is not None:
                elapsed = round(time.time() - t0)
                tokens_used = int(CompletionTokens.instance().value - tokens_before)
                agent_name = self.nodes[current_id].agent_name
                if agent_name in self.execution_trace.trace[-1]:
                    self.execution_trace.trace[-1][agent_name]["time"] = elapsed
                    self.execution_trace.trace[-1][agent_name]["completion_tokens"] = tokens_used
                self.execution_trace.trace[-1]["exec_order"].append(agent_name)

            # Unlock successors.
            for succ in self.nodes[current_id].spatial_successors:
                if succ.id not in active_ids:
                    continue
                in_degree[succ.id] = max(in_degree.get(succ.id, 1) - 1, 0)
                if in_degree[succ.id] == 0 and succ.id not in executed:
                    queue.append(succ.id)

        pbar.close()

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _build_topology(self, topology: RoundTopology, active_agents: Set[str]) -> None:
        """Wire spatial edges for this round based on the topology definition.

        Only edges where both endpoints are in active_agents are created.
        """
        self._clear_spatial()
        for sender_name, receiver_name in topology.edges:
            sender = self._get_node_by_name(sender_name)
            receiver = (
                self.nodes.get(self.collector_id)
                if receiver_name == "Collector"
                else self._get_node_by_name(receiver_name)
            )

            if sender is None or receiver is None:
                continue

            sender_active = sender.agent_name in active_agents
            receiver_active = receiver.agent_name in active_agents

            if sender_active and receiver_active:
                sender.add_successor(receiver, "spatial")

    def _connect_temporal(self) -> None:
        """Connect temporal self-edges for nodes with prior-round outputs.

        Skipped entirely during DRAFTING: every round targets a fresh section,
        so a prior-round output from a different section contaminates context
        rather than helping. All durable cross-round state (section list,
        current directive, written prose) is carried by ReportState, making
        temporal self-edges redundant and actively harmful in this phase.
        """
        self._clear_temporal()
        if self.phase_state.current_phase == PhaseType.DRAFTING:
            return
        for node in self.nodes.values():
            if node.last_memory["outputs"]:
                node.add_predecessor(node, "temporal")

    def _clear_spatial(self) -> None:
        for node in self.nodes.values():
            node.spatial_predecessors = []
            node.spatial_successors = []

    def _clear_temporal(self) -> None:
        for node in self.nodes.values():
            node.temporal_predecessors = []
            node.temporal_successors = []

    def _clear_all_memory(self) -> None:
        """Wipe all per-node state at a phase or section boundary.

        Clears both last_memory AND the live output/input buffers. Clearing
        only last_memory is insufficient: _update_memory() copies node.outputs
        into last_memory after every round, so a node that did not execute in
        the current round would carry stale outputs from a previous phase or
        section into last_memory — making TEMPORAL_HEURISTIC falsely include it
        in subsequent rounds. Zeroing node.outputs/inputs/raw_inputs here
        ensures that non-executing nodes produce empty last_memory entries.
        """
        for node in self.nodes.values():
            node.last_memory = {"inputs": [], "outputs": [], "raw_inputs": []}
            node.outputs = []
            node.inputs = []
            node.raw_inputs = []

    def _update_memory(self) -> None:
        for node in self.nodes.values():
            node.update_memory()

    # ------------------------------------------------------------------
    # Execution trace helpers
    # ------------------------------------------------------------------

    def _init_trace_round(self) -> None:
        """Append a fresh round slot to the execution trace.

        Schema mirrors QMIXGraph so StandaloneVisualizer works without changes:
          - One entry per agent name (action=None for handcrafted runs)
          - "RAG" entry (populated later by Researcher agent)
          - "Collector" entry with report_state snapshot
          - "exec_order" list (populated as agents execute)
        """
        round_data: Dict[str, Any] = {
            name: {
                "action": None,
                "message_to": [],
                "prompt": None,
                "response": None,
                "time": None,
                "completion_tokens": None,
            }
            for name in self.agent_names
        }
        round_data["RAG"] = {"action": None, "message_to": [], "prompt": None, "response": None, "sources": []}
        if self.collector_id is not None:
            round_data["Collector"]["report_state"] = ReportState.instance().content
        round_data["exec_order"] = []
        self.execution_trace.trace.append(round_data)

    def _trace_spatial_edges(self) -> None:
        """Pre-populate message_to from already-built spatial edges.

        Called after _build_topology so the visualizer can draw arrows even
        for agents whose prompt/response haven't been recorded yet.
        RAG↔Researcher links are written inside researcher.py and are skipped here.
        """
        for node in self.nodes.values():
            agent_name = node.agent_name
            if agent_name not in self.execution_trace.trace[-1]:
                continue
            for succ in node.spatial_successors:
                if succ.agent_name in self.execution_trace.trace[-1]:
                    self.execution_trace.trace[-1][agent_name]["message_to"].append(succ.agent_name)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Validation directive decomposition
    # ------------------------------------------------------------------

    async def _decompose_validation_directive(
        self,
        combined_issues: str,
        report_state,
    ) -> str:
        """Ask the LLM to decompose global validation issues into per-section actions.

        Returns a bulleted list of `- section_N: <action>` items, or the raw
        combined_issues string if the LLM call fails.
        """
        llm = self._get_any_llm()
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
            logger.warning(f"[{self.id}] Directive decomposition failed: {exc}")
            return combined_issues

    async def _generate_abstract(self, report_state) -> str:
        """Write a concise reader-facing abstract for the finished report.

        Runs once at assembly time over the final body content, so it reflects
        the report after all validation/revision is complete. Returns the
        abstract paragraph (no heading), or "" if no body exists or the call
        fails — callers must treat "" as "skip the abstract".
        """
        body = (report_state.content or "").strip()
        if not body:
            return ""

        llm = self._get_any_llm()
        if llm is None:
            return ""

        system = (
            "You are a scientific editor writing the abstract for a completed "
            "technical report. Summarise the report for a reader deciding whether "
            "to read it.\n\n"
            "RULES:\n"
            "1. Write ONE self-contained paragraph of roughly 150-250 words.\n"
            "2. Cover the report's scope/objective, the approach or evidence it "
            "draws on, its key findings, and its main conclusions — in that order.\n"
            "3. Plain expository prose. Do NOT include a heading, a 'In this report' "
            "preamble, bullet points, or section references of any kind.\n"
            "4. Do NOT include citation tags (e.g. [cite:3]), bibliography numbers, "
            "or figure/section labels. The abstract must read as standalone prose.\n"
            "5. Use only information present in the report body below — do not "
            "introduce claims, numbers, or conclusions that are not in the text."
        )
        user = (
            f"### Report subject\n{report_state.task}\n\n"
            f"### Report body\n{body}\n\n"
            "Write the abstract now. Output ONLY the abstract paragraph, with no "
            "heading and no surrounding commentary."
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            response = await llm.agen(message, calling_agent="LeadArchitect")
        except Exception as exc:
            logger.warning(f"[{self.id}] Abstract generation failed: {exc}")
            return ""

        abstract = (response or "").strip()
        if not abstract:
            return ""

        # Strip a leading "Abstract"/"## Abstract" heading if the model added one
        # despite the instruction (we supply the heading ourselves at assembly).
        abstract = re.sub(r"(?i)^\s*#*\s*abstract\s*[:\-]?\s*\n+", "", abstract).strip()
        # Remove any stray citation tags so the abstract stays self-contained.
        abstract = re.sub(r"\s*\[cite:[^\]]*\]", "", abstract).strip()
        return abstract

    # ------------------------------------------------------------------
    # Bibliography
    # ------------------------------------------------------------------

    # Matches new arXiv format (e.g. 2605.30554) and old format (e.g. 0208016 / 9804027).
    _ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$", re.IGNORECASE)
    _ARXIV_OLD_RE = re.compile(r"^(\d{7})(v\d+)?\.pdf$", re.IGNORECASE)

    def _build_bibliography(self) -> None:
        """Build the final bibliography text using per-entry citation counts.

        bibliography_map is populated incrementally by _apply_citation_tags —
        only sources that were actually cited in the text have entries there.
        All other collected sources are listed under Consulted Sources (no tag).

        Must be called after all _apply_citation_tags() calls.
        """
        report_state = ReportState.instance()
        bib_map = report_state.bibliography_map
        counts  = report_state.citation_counts

        if not report_state.sources:
            logger.warning(f"[{self.id}] Bibliography: no sources collected — skipping.")
            return

        # Map each source filename to the bibliographic metadata (title/author/year)
        # carried on its retrieved chunks. First non-empty wins per source.
        meta_by_source: dict = {}
        for doc in report_state.sources:
            src = (doc.get("source") or "").strip()
            if not src or src in meta_by_source:
                continue
            fields = {k: (doc.get(k) or "").strip() for k in ("title", "author", "year")}
            if any(fields.values()):
                meta_by_source[src] = fields

        cited_lines: list = ["## Bibliography\n"]
        for source_name, num in sorted(bib_map.items(), key=lambda x: x[1]):
            c = counts.get(num, 0)
            count_tag = f" *({c} citation{'s' if c != 1 else ''})*"
            cited_lines.append(
                self._format_bib_entry(num, source_name, meta_by_source.get(source_name)) + count_tag
            )

        seen: set = set()
        consulted_lines: list = []
        for doc in report_state.sources:
            src = (doc.get("source") or "").strip()
            if src and src not in bib_map and src not in seen:
                seen.add(src)
                consulted_lines.append(
                    self._format_consulted_entry(src, meta_by_source.get(src))
                )

        report_state.bibliography = "\n".join(cited_lines)
        if consulted_lines:
            report_state.bibliography += (
                "\n\n### Consulted Sources\n\n" + "\n".join(consulted_lines)
            )

        logger.info(
            f"[{self.id}] Bibliography built: "
            f"{len(cited_lines) - 1} cited, {len(consulted_lines)} consulted-only."
        )

    @classmethod
    def _arxiv_id_of(cls, source_name: str) -> str:
        """Return the arXiv id encoded in the filename, or '' if it is not an arXiv PDF."""
        m = cls._ARXIV_NEW_RE.match(source_name) or cls._ARXIV_OLD_RE.match(source_name)
        return m.group(1) if m else ""

    @classmethod
    def _compose_reference(cls, source_name: str, meta: Optional[dict]) -> str:
        """Build a proper reference string from available metadata.

        Renders "Author. “Title”. Year. (identifier)" using whatever fields are
        present, always keeping the source filename as a locator so [cite:N] tags
        remain traceable. Falls back to a filename-first entry when no
        bibliographic metadata was extracted for the document.
        """
        meta = meta or {}
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        year = (meta.get("year") or "").strip()
        arxiv_id = cls._arxiv_id_of(source_name)

        # Identifier suffix: arXiv id (when present) plus the filename locator.
        ident_bits = []
        if arxiv_id:
            ident_bits.append(f"arXiv:{arxiv_id}")
        ident_bits.append(source_name)
        ident = ", ".join(ident_bits)

        parts = []
        if author:
            parts.append(author if author.endswith(".") else f"{author}.")
        if title:
            # Curly quotes become proper LaTeX quotes after markdown conversion.
            parts.append(f"“{title}”.")
        if year:
            parts.append(f"{year}.")

        if parts:
            return " ".join(parts) + f" ({ident})"

        # No descriptive metadata — keep the legacy filename-first rendering.
        if arxiv_id:
            return f"**{source_name}** *(arXiv:{arxiv_id})*"
        ext = source_name.rsplit(".", 1)[-1].upper() if "." in source_name else ""
        type_tag = f" *({ext})*" if ext else ""
        return f"**{source_name}**{type_tag}"

    @classmethod
    def _format_bib_entry(cls, num: int, source_name: str, meta: Optional[dict] = None) -> str:
        """Return one markdown bibliography line for the given source."""
        return f"[{num}] " + cls._compose_reference(source_name, meta)

    @classmethod
    def _format_consulted_entry(cls, source_name: str, meta: Optional[dict] = None) -> str:
        """Return one markdown consulted-sources line (no citation number)."""
        return "- " + cls._compose_reference(source_name, meta)

    # ------------------------------------------------------------------
    # Citation tagging
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> frozenset:
        """Return a frozenset of meaningful tokens from text.

        Keeps alphanumeric tokens of length >= _MIN_TOKEN_LEN that are not in
        _STOPWORDS.  Numbers are kept at length >= 3 so that values like "154"
        or "360" contribute to overlap scoring alongside longer word tokens.
        """
        result = set()
        for t in _TOKEN_RE.findall(text.lower()):
            if t in _STOPWORDS:
                continue
            # Numbers: keep at >= 3 chars; words: keep at >= _MIN_TOKEN_LEN
            if t.isdigit():
                if len(t) >= 3:
                    result.add(t)
            elif len(t) >= _MIN_TOKEN_LEN:
                result.add(t)
        return frozenset(result)

    def _apply_citation_tags(self, section_idx: int) -> None:
        """Insert citation tags into a section using sentence-level n-gram overlap.

        For each non-heading line in the section, sentences are split on
        punctuation boundaries.  Each sentence is scored against the RAG chunks
        that were retrieved when the section was drafted.  When the shared
        meaningful-token count meets _MIN_CITATION_OVERLAP, a [cite:N, p.X] tag
        is appended after the sentence (or [cite:N] if no page info is stored).

        The updated content is written back via replace_section() so that the
        sources list for the section is preserved unchanged.
        """
        report_state = ReportState.instance()
        sections = report_state.sections
        if not (0 <= section_idx < len(sections)):
            return

        section = sections[section_idx]
        sources = section.get("sources", [])
        bib_map = report_state.bibliography_map  # mutated in-place on first citation

        # Known source filenames (global, case-insensitive) used to resolve
        # inline references. Includes sources already numbered in earlier sections.
        known_lc: Dict[str, str] = {}
        for doc in report_state.sources:
            s = doc.get("source", "").strip()
            if s:
                known_lc.setdefault(s.lower(), s)
        for s in bib_map:
            known_lc.setdefault(s.lower(), s)

        # Pre-tokenize each source chunk; bib numbers are assigned lazily.
        chunk_refs: List[Tuple[frozenset, str, Optional[str]]] = []
        for doc in sources:
            src = doc.get("source", "").strip()
            if not src:
                continue
            page = doc.get("page")
            page_str = str(page) if page and str(page) != "N/A" else None
            chunk_refs.append((self._tokenize(doc.get("content", "")), src, page_str))

        # Nothing to do if we can neither score overlap nor resolve inline refs.
        if not chunk_refs and not known_lc:
            return

        new_lines: List[str] = []
        tagged_count = 0
        inline_count = 0
        orphan_count = 0

        for line in section["content"].split("\n"):
            # Preserve headings and blank lines as-is.
            if not line.strip() or line.lstrip().startswith("#"):
                new_lines.append(line)
                continue

            # Split into (sentence, separator, sentence, separator, …) preserving
            # the inter-sentence whitespace via the capturing group in the regex.
            parts = _SENTENCE_SPLIT_RE.split(line)
            new_parts: List[str] = []

            for j, part in enumerate(parts):
                # Odd-indexed parts are the captured whitespace separators.
                if j % 2 != 0:
                    new_parts.append(part)
                    continue

                # First rewrite any inline source references the model wrote in
                # this sentence into proper [cite:N, p.X] tags.
                if known_lc:
                    part, n_inline = self._rewrite_inline_references(
                        part, known_lc, bib_map, report_state
                    )
                    inline_count += n_inline

                # Then run overlap-based tagging on the (possibly updated) sentence.
                sent_tok = self._tokenize(part) if chunk_refs else frozenset()
                if chunk_refs and len(sent_tok) >= _MIN_SENTENCE_TOKENS:
                    # Collect matching pages grouped by source name.
                    pages_by_src: Dict[str, List[str]] = {}
                    for chunk_tok, src_name, page_str in chunk_refs:
                        if len(sent_tok & chunk_tok) >= _MIN_CITATION_OVERLAP:
                            if src_name not in pages_by_src:
                                pages_by_src[src_name] = []
                            if page_str and page_str not in pages_by_src[src_name]:
                                pages_by_src[src_name].append(page_str)

                    # Assign bib numbers lazily on first citation, then sort by number.
                    for src_name in pages_by_src:
                        if src_name not in bib_map:
                            bib_map[src_name] = len(bib_map) + 1

                    tags: List[str] = []
                    for src_name, pages in sorted(pages_by_src.items(), key=lambda x: bib_map[x[0]]):
                        bib_num = bib_map[src_name]
                        if not pages:
                            tag = f"[cite:{bib_num}]"
                        elif len(pages) == 1:
                            tag = f"[cite:{bib_num}, p.{pages[0]}]"
                        else:
                            tag = f"[cite:{bib_num}, pp.{','.join(pages)}]"
                        # Skip if this document is already cited in this sentence.
                        if not re.search(rf'\[cite:{bib_num}[,\]]', part):
                            tags.append(tag)
                            report_state.citation_counts[bib_num] = (
                                report_state.citation_counts.get(bib_num, 0) + 1
                            )

                    if tags:
                        # Insert tags before the trailing sentence-ending punctuation.
                        trailing_m = re.search(r'([.!?])\s*$', part)
                        if trailing_m:
                            pos = trailing_m.start()
                            part = part[:pos] + " " + " ".join(tags) + part[pos:]
                        else:
                            part = part.rstrip() + " " + " ".join(tags)
                        tagged_count += len(tags)

                # Finally drop stray numeric reference markers (e.g. "Ref. [32]")
                # the model copied from a source chunk that don't correspond to a
                # [cite:N] tag applied to this sentence.
                part, n_orphan = self._strip_orphan_citation_markers(part)
                orphan_count += n_orphan
                new_parts.append(part)

            new_lines.append("".join(new_parts))

        if tagged_count > 0 or inline_count > 0 or orphan_count > 0:
            report_state.replace_section(section["id"], "\n".join(new_lines))
            logger.info(
                f"[{self.id}] Section {section_idx + 1}: "
                f"{tagged_count} citation tag(s) inserted, "
                f"{inline_count} inline reference(s) rewritten, "
                f"{orphan_count} stray marker(s) removed."
            )
        else:
            logger.info(
                f"[{self.id}] Section {section_idx + 1}: "
                f"no citation matches above threshold — section unchanged."
            )

    def _rewrite_inline_references(
        self,
        text: str,
        known_lc: Dict[str, str],
        bib_map: Dict[str, int],
        report_state,
    ) -> Tuple[str, int]:
        """Rewrite inline source references in one sentence into citation tags.

        Handles bracketed forms (``[file.pdf | Page: 26]`` and the
        ``[Source: file.pdf | Page: 26]`` variant with an inline ``Source:``
        prefix), ``Source file.pdf (Page 2)`` prose, and bare ``(file.pdf)``
        mentions. Only references whose
        filename resolves to a known source are rewritten; unknown filenames are
        left untouched. Bib numbers are assigned lazily (matching the overlap
        pass) and ``citation_counts`` is updated per new tag. At most one tag is
        kept per source per sentence — the same rule the overlap pass enforces —
        so a source already cited in the sentence (by an existing tag, a prior
        inline reference, or one the overlap pass will add) is never duplicated;
        the redundant inline reference text is dropped instead.

        Returns ``(new_text, n_changes)``.
        """
        # Bib numbers already cited in this sentence. Seeded from any tags
        # already present and grown as we rewrite, so we never emit two tags for
        # the same source. \d+ captures the full number, so "1" never matches
        # inside "[cite:12]".
        present_nums = set(re.findall(r'\[cite:(\d+)', text))
        n = 0

        def _build_tag(num: int, pages: List[str]) -> str:
            if not pages:
                return f"[cite:{num}]"
            if len(pages) == 1:
                return f"[cite:{num}, p.{pages[0]}]"
            return f"[cite:{num}, pp.{','.join(pages)}]"

        def _repl(m: "re.Match") -> str:
            nonlocal n
            src = known_lc.get(m.group(1).strip().lower())
            if src is None:
                return m.group(0)  # unknown filename — leave the prose as written
            pages = re.findall(r'\d+', m.group(2)) if m.re.groups >= 2 and m.group(2) else []
            if src not in bib_map:
                bib_map[src] = len(bib_map) + 1
            num = bib_map[src]
            n += 1
            if str(num) in present_nums:
                return ""  # source already cited in this sentence — drop reference
            present_nums.add(str(num))
            report_state.citation_counts[num] = (
                report_state.citation_counts.get(num, 0) + 1
            )
            return _build_tag(num, pages)

        out = _INLINE_REF_BRACKET_RE.sub(_repl, text)
        out = _INLINE_REF_SOURCE_RE.sub(_repl, out)
        out = _INLINE_REF_PAREN_RE.sub(_repl, out)

        if n:
            # Tidy whitespace / empty parens left by removed references.
            out = re.sub(r'\(\s*\)', '', out)
            out = re.sub(r'[ \t]{2,}', ' ', out)
            out = re.sub(r'\s+([.,;:])', r'\1', out)
        return out, n

    def _strip_orphan_citation_markers(self, text: str) -> Tuple[str, int]:
        """Remove stray bare numeric reference markers from a sentence.

        The model occasionally copies a bracketed reference marker straight out
        of a retrieved chunk (e.g. ``Ref. [32]``), where the number indexes that
        source's own reference list rather than this report's bibliography. The
        report only ever cites sources with ``[cite:N, p.X]`` tags, so a bare
        ``[N]`` marker is never a valid citation — it is always dropped to keep a
        consistent format and avoid duplicated references. Real ``[cite:N, p.X]``
        tags are not matched by the marker pattern, so they are left untouched.

        Returns ``(new_text, n_removed)``.
        """
        n = 0

        def _repl(m: "re.Match") -> str:
            nonlocal n
            n += 1
            return ""

        out = _ORPHAN_CITE_RE.sub(_repl, text)
        if n:
            # Tidy whitespace left where markers were removed.
            out = re.sub(r'[ \t]{2,}', ' ', out)
            out = re.sub(r'\s+([.,;:])', r'\1', out)
        return out, n

    def _get_node_by_name(self, agent_name: str) -> Optional[Node]:
        for node in self.nodes.values():
            if node.agent_name == agent_name:
                return node
        return None

    def _get_any_llm(self):
        """Return the LLM from the first node that has one (for gatecheck calls)."""
        for node in self.nodes.values():
            if hasattr(node, "llm"):
                return node.llm
        return None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(n.spatial_successors) for n in self.nodes.values())
