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
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import shortuuid
from tqdm import tqdm

from graph.node import Node
from handcrafted_graph.phases import PhaseConfig, PhaseType, RoundTopology, PHASE_SEQUENCE
from handcrafted_graph.scheduler import RoundScheduler, SkipStrategy
from handcrafted_graph.state import PhaseState
from utils.globals import PromptTokens, CompletionTokens, ReportState, ExecutionTrace

logger = logging.getLogger("handcrafted_graph")


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

        # ── Bar 2: Correction (SECTION_REVIEW → VALIDATION) ──────────────
        # Total is now exact: we know n_sections, so no patching at runtime.
        correction_total = sum(
            p.max_rounds * n_sections if p.section_aware else p.max_rounds
            for p in correction_phases
        )
        correction_pbar = tqdm(
            total=correction_total, desc="Review & correction", unit="round", leave=True
        )
        try:
            for phase in correction_phases:
                self.phase_state.set_phase(phase.name)
                logger.info(f"[{self.id}] Starting phase: {phase.name.value.upper()}")
                await self._execute_phase(input, phase, max_tries, max_time, correction_pbar)
        finally:
            correction_pbar.close()

        report = ReportState.instance().content
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
            if "[NO_REVISION_NEEDED]" in reviewer_output:
                logger.info(
                    f"  [{phase.name.value}] Section {i + 1}/{n_sections}: "
                    f"no revision needed — skipping."
                )
                if overall_pbar is not None:
                    overall_pbar.update(1)  # account for skipped revision round
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
        """Wipe last_memory for every node at a phase boundary.

        Prevents stale temporal outputs from a finished phase from
        pattern-biasing agent behaviour in the next one. All durable
        cross-phase state (section list, current directive, progress)
        is carried by ReportState, not by temporal memory.
        """
        for node in self.nodes.values():
            node.last_memory = {"inputs": [], "outputs": [], "raw_inputs": []}

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
        round_data["RAG"] = {"action": None, "message_to": [], "prompt": None, "response": None}
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
