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
from utils.globals import PromptTokens, CompletionTokens, ReportState, ExecutionTrace, SourceBuffer

logger = logging.getLogger("handcrafted_graph")

# ------------------------------------------------------------------
# Citation tagging — sentence-level n-gram overlap constants
# ------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(\s+)(?=[A-Z"\'])')
_TOKEN_RE          = re.compile(r'[a-zA-Z0-9]+')
_MIN_TOKEN_LEN        = 3   # minimum token length to be considered meaningful
_MIN_CITATION_OVERLAP = 3   # shared tokens required to assign a citation
_MIN_SENTENCE_TOKENS  = 3   # sentence must have this many tokens to be a candidate

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
        skip_strategy: SkipStrategy = SkipStrategy.TEMPORAL_HEURISTIC,
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
        max_validation_attempts: int = 2,
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
        finally:
            writing_pbar.close()

        # ── Transition notice ─────────────────────────────────────────────
        n_sections = len(ReportState.instance().sections)
        tqdm.write(
            f"\n  Writing complete — {n_sections} section(s) drafted."
            f"  Starting review & correction phase…\n"
        )

        # ── Bibliography — built once; guard prevents rebuild on validation retries ──
        report_state = ReportState.instance()
        if not report_state.bibliography_map:
            self._build_bibliography()

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
                    windows = self._build_section_windows(
                        sections, p.window_size, p.window_overlap_sections
                    )
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
        report = report_state.content
        if report_state.bibliography:
            report = report.rstrip() + "\n\n" + report_state.bibliography

        if self.execution_trace is not None:
            self.execution_trace.trace[-1]["Collector"]["report_state"] = report

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

            if ReportState.instance().task == "[DRAFTING_COMPLETE]":
                logger.info(
                    f"Phase '{phase.name.value}' completed early at round {round_idx}: "
                    f"[DRAFTING_COMPLETE] signal received."
                )
                if overall_pbar is not None:
                    overall_pbar.update(phase.max_rounds - round_idx - 1)
                ReportState.instance().task = "[WAITING FOR NEXT PHASE DIRECTIVE]"
                break

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
        sections = report_state.sections
        n_sections = len(sections)

        if n_sections == 0:
            logger.warning(f"Phase '{phase.name.value}': no sections to review — skipping.")
            return

        review_topology = phase.round_topologies[0]
        revision_topology = phase.round_topologies[1]

        for i, section in enumerate(sections):
            report_state.review_section_idx = i
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
            _without_signal = reviewer_output.replace(_signal, "").strip().strip("-").strip()
            _is_clean_pass = (
                _signal in reviewer_output
                and not _CORRECTION_RE.search(_without_signal)
            )
            if _is_clean_pass:
                logger.info(
                    f"  [{phase.name.value}] Section {i + 1}/{n_sections}: "
                    f"no revision needed — skipping."
                )
                if overall_pbar is not None:
                    overall_pbar.update(1)  # account for skipped revision round
                self._apply_citation_tags(i)
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

            active_agents = await scheduler.get_active_agents(
                revision_topology, 1, task_input=input
            )
            logger.info(
                f"  [{phase.name.value}] Section {i + 1}/{n_sections} revision: "
                f"active={sorted(active_agents)}"
            )

            self._build_topology(revision_topology, active_agents)
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

            self._apply_citation_tags(i)
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
            i += max(1, len(window) - overlap_sections)
        return windows

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
        sections = report_state.sections
        windows = self._build_section_windows(
            sections, phase.window_size, phase.window_overlap_sections
        )
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
            "4. NEVER write 'because section_X says' or 'consistent with section_X' or 'as per "
            "section_Y' — each instruction must stand alone. Reference only physical facts, "
            "observational evidence, or source citations, never other section IDs."
        )
        section_list = report_state.list_sections(verbose=True)
        user = (
            f"### Report sections\n{section_list}\n\n"
            f"### Identified issues\n{combined_issues}\n\n"
            "Output ONLY a bulleted list using this exact format:\n"
            "  - <section_id>: <specific action with exact value if applicable>\n\n"
            "One bullet per section that needs changing. If the same factual value must appear in "
            "multiple sections, list each section separately and give EACH a distinct angle or "
            "sub-topic so they do not duplicate each other. "
            "Use exact section IDs from the list above. Skip praise or general observations."
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

    # ------------------------------------------------------------------
    # Bibliography
    # ------------------------------------------------------------------

    # Matches new arXiv format (e.g. 2605.30554) and old format (e.g. 0208016 / 9804027).
    _ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$", re.IGNORECASE)
    _ARXIV_OLD_RE = re.compile(r"^(\d{7})(v\d+)?\.pdf$", re.IGNORECASE)

    def _build_bibliography(self) -> None:
        """Build a deduplicated markdown bibliography from all collected sources.

        Deduplicates by source filename, preserving first-encounter order
        (PLANNING → RESEARCH → DRAFTING sections).  Populates:
          - ReportState.bibliography      : the markdown text
          - ReportState.bibliography_map  : {source_filename: bib_number}

        No LLM calls — pure metadata extraction from already-collected sources.
        """
        report_state = ReportState.instance()
        all_sources = report_state.sources

        if not all_sources:
            logger.warning(f"[{self.id}] Bibliography: no sources collected — skipping.")
            return

        # Deduplicate by source filename, keep first-seen doc dict for metadata.
        seen: Dict[str, dict] = {}
        for doc in all_sources:
            src = doc.get("source", "").strip()
            if src and src not in seen:
                seen[src] = doc

        # Build mapping and markdown in one pass.
        bib_map: Dict[str, int] = {}
        lines = ["## Bibliography\n"]

        for num, (source_name, doc) in enumerate(seen.items(), start=1):
            bib_map[source_name] = num
            lines.append(self._format_bib_entry(num, source_name))

        report_state.bibliography_map = bib_map
        report_state.bibliography = "\n".join(lines)
        logger.info(
            f"[{self.id}] Bibliography built: {len(seen)} unique source(s)."
        )

    @classmethod
    def _format_bib_entry(cls, num: int, source_name: str) -> str:
        """Return one markdown bibliography line for the given source filename."""
        # Try new arXiv format first (e.g. 2605.30554.pdf)
        m = cls._ARXIV_NEW_RE.match(source_name)
        if m:
            arxiv_id = m.group(1)
            return f"[{num}] **{source_name}** *(arXiv:{arxiv_id})*"

        # Try old arXiv format (e.g. 0208016.pdf — 7-digit hep-*/astro-ph IDs)
        m = cls._ARXIV_OLD_RE.match(source_name)
        if m:
            arxiv_id = m.group(1)
            return f"[{num}] **{source_name}** *(arXiv:{arxiv_id})*"

        # Generic fallback: just the filename, file type noted
        ext = source_name.rsplit(".", 1)[-1].upper() if "." in source_name else ""
        type_tag = f" *({ext})*" if ext else ""
        return f"[{num}] **{source_name}**{type_tag}"

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
        bib_map = report_state.bibliography_map

        if not sources or not bib_map:
            return

        # Pre-tokenize each source chunk and resolve its bibliography number.
        chunk_refs: List[Tuple[frozenset, int, Optional[str]]] = []
        for doc in sources:
            src = doc.get("source", "").strip()
            bib_num = bib_map.get(src)
            if bib_num is None:
                continue
            page = doc.get("page")
            page_str = str(page) if page and str(page) != "N/A" else None
            chunk_refs.append((self._tokenize(doc.get("content", "")), bib_num, page_str))

        if not chunk_refs:
            return

        new_lines: List[str] = []
        tagged_count = 0

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

                sent_tok = self._tokenize(part)
                if len(sent_tok) < _MIN_SENTENCE_TOKENS:
                    new_parts.append(part)
                    continue

                # Collect matching pages grouped by bibliography number.
                pages_by_num: Dict[int, List[str]] = {}
                for chunk_tok, bib_num, page_str in chunk_refs:
                    if len(sent_tok & chunk_tok) >= _MIN_CITATION_OVERLAP:
                        if bib_num not in pages_by_num:
                            pages_by_num[bib_num] = []
                        if page_str and page_str not in pages_by_num[bib_num]:
                            pages_by_num[bib_num].append(page_str)

                tags: List[str] = []
                for bib_num in sorted(pages_by_num.keys()):
                    pages = pages_by_num[bib_num]
                    if not pages:
                        tag = f"[cite:{bib_num}]"
                    elif len(pages) == 1:
                        tag = f"[cite:{bib_num}, p.{pages[0]}]"
                    else:
                        tag = f"[cite:{bib_num}, pp.{','.join(pages)}]"
                    # Skip if this document is already cited in this sentence.
                    if not re.search(rf'\[cite:{bib_num}[,\]]', part):
                        tags.append(tag)

                if tags:
                    # Insert tags before the trailing sentence-ending punctuation.
                    trailing_m = re.search(r'([.!?])\s*$', part)
                    if trailing_m:
                        pos = trailing_m.start()
                        new_parts.append(
                            part[:pos] + " " + " ".join(tags) + part[pos:]
                        )
                    else:
                        new_parts.append(part.rstrip() + " " + " ".join(tags))
                    tagged_count += len(tags)
                else:
                    new_parts.append(part)

            new_lines.append("".join(new_parts))

        if tagged_count > 0:
            report_state.replace_section(section["id"], "\n".join(new_lines))
            logger.info(
                f"[{self.id}] Section {section_idx + 1}: "
                f"{tagged_count} citation tag(s) inserted."
            )
        else:
            logger.info(
                f"[{self.id}] Section {section_idx + 1}: "
                f"no citation matches above threshold — section unchanged."
            )

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
