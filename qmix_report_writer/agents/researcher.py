import asyncio
import re

from qmix_report_writer.graph.node import Node
from qmix_report_writer.agents.agent_registry import AgentRegistry
from qmix_report_writer.tools.rag import RAGManager
from qmix_report_writer.utils.config import get_llm, get_rag_config
from qmix_report_writer.utils.globals import ReportState, SourceBuffer
from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry

_DEFICIENCY_RE = re.compile(r"State Deficiency", re.IGNORECASE)
_QUERY_PREFIX_RE = re.compile(r"^[\d\w]+[.)]\s*")


def _parse_queries(raw: str) -> list:
    """Extract up to 3 valid search queries from a multi-line LLM response.

    Strips citation artefacts (| [source: ...]) that leak from the Researcher's
    evidence format into the query formulation context, then rejects lines that
    are clearly natural-language prose rather than keyword search strings via:
      - word count > 15: catches long sentences beyond the 8-term prompt limit
      - sentence boundary ('. ' or trailing '.'): catches shorter prose
    """
    queries = []
    for line in raw.strip().splitlines():
        line = _QUERY_PREFIX_RE.sub("", line.strip())
        # Strip pipe-separated citation suffixes: "... | [source: file.pdf]"
        line = re.sub(r"\s+\|.*$", "", line).strip()
        if (line
                and not line.startswith("[")
                and line.upper() != "NO_QUERY"
                and len(line.split()) <= 15
                and ". " not in line
                and not line.endswith(".")):
            queries.append(line)
    return queries[:3]


@AgentRegistry.register("Researcher")
class Researcher(Node):
    """Main source of information of the team, here to find documents and citations."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Researcher", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Researcher"
        self.report = ReportState.instance()
        rag_cfg = get_rag_config()
        rerank_mode = rag_cfg.get("rerank_mode", "nomic")
        bm25_floor = int(rag_cfg.get("bm25_floor", 1))
        self.top_k_planning = int(rag_cfg.get("top_k_planning", 3))
        self.top_k_drafting = int(rag_cfg.get("top_k_drafting", 5))
        self.rag = RAGManager(rerank_mode=rerank_mode, bm25_floor=bm25_floor)
        # Chunk ids already reported during RESEARCH. Consecutive research rounds
        # tend to issue near-identical queries; without this filter the same
        # chunks are re-retrieved and re-synthesised round after round.
        self._reported_chunk_ids: set = set()

        # Optional PBDS parameter-dependency enrichment. Activates only when a
        # workbook is configured and present; any load/parse problem disables it
        # silently so the pipeline is unchanged whenever the tool is off.
        self._pbds_manager = None
        self._pbds_matcher = None
        # Per-run frontier of parameters already surfaced by the tool. Used in
        # PLANNING/RESEARCH so documents later retrieved FOR a connected parameter
        # do not re-trigger expansion (keeps research covering the subject rather
        # than walking the dependency graph).
        self._pbds_surfaced_nodes: set = set()
        try:
            from qmix_report_writer.tools.pbds import NodeMatcher, load_pbds_manager
            manager = load_pbds_manager()
            if manager is not None:
                self._pbds_matcher = NodeMatcher(manager)  # builds the graph/BM25 index
                self._pbds_manager = manager
        except Exception:
            self._pbds_manager = None
            self._pbds_matcher = None

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, pbds_block=None, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Current report state", self.report.progress,
            pbds_block=pbds_block,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _is_revision_phase(self) -> bool:
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.SECTION_REVIEW
        except Exception:
            return False

    def _current_top_k(self) -> int:
        """Return top_k appropriate for the active phase.

        PLANNING uses a lower value — the goal is topic discovery, not evidence depth.
        All other phases (RESEARCH, DRAFTING, SECTION_REVIEW) use the drafting value.
        """
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            if PhaseState.instance().current_phase == PhaseType.PLANNING:
                return self.top_k_planning
        except Exception:
            pass
        return self.top_k_drafting

    def _is_planning_phase(self) -> bool:
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.PLANNING
        except Exception:
            return False

    def _is_research_phase(self) -> bool:
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.RESEARCH
        except Exception:
            return False

    # Returned (without an LLM call) when every retrieved chunk was already
    # reported in an earlier RESEARCH round. Phrased as a State Deficiency so
    # the LeadArchitect redirects the next query instead of ending the phase.
    _DUPLICATE_RETRIEVAL_SIGNAL = (
        "State Deficiency: all retrieved documents were already reported in earlier "
        "research rounds. Redirect the next query to a different sub-topic."
    )

    def _filter_reported_chunks(self, documents: list) -> list:
        """Drop chunks already reported in a previous RESEARCH round.

        Applies only to RESEARCH — DRAFTING and SECTION_REVIEW legitimately
        re-retrieve chunks because each section keeps its own source list for
        fact-checking and citation tagging.
        """
        fresh = [d for d in documents if d.get("id") not in self._reported_chunk_ids]
        self._reported_chunk_ids.update(d.get("id") for d in fresh)
        return fresh

    def _persist_deficiencies(self, response: str) -> None:
        """Parse State Deficiency entries from a PLANNING coverage response and store them."""
        for match in re.finditer(r"State Deficiency:\s*(.+?)(?:\n|$)", response, re.IGNORECASE):
            self.report.add_deficiency(match.group(1).strip())

    def _get_da_output(self, spatial_info: dict):
        """Return DataAnalyst's message from spatial_info, or None if absent."""
        for info in spatial_info.values():
            if info.get("role") == "Data Analyst":
                return str(info.get("output", ""))
        return None

    # ------------------------------------------------------------------ #
    # PBDS parameter-dependency enrichment (optional; no-op when the tool is off)
    # ------------------------------------------------------------------ #
    def _pbds_phase(self):
        """Return the active phase when PBDS should run (PLANNING/RESEARCH/DRAFTING), else None."""
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            phase = PhaseState.instance().current_phase
            if phase in (PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING):
                return phase
        except Exception:
            pass
        return None

    def _pbds_label(self, node: str) -> str:
        """Human-readable label for a node (its description, or the raw name)."""
        return self._pbds_manager.descriptions().get(node) or node

    def _pbds_format_block(self, findings, phase) -> str:
        """Render the phase-appropriate dependency-graph block from (Candidate, neighborhood) findings."""
        from qmix_report_writer.handcrafted_graph.phases import PhaseType
        lines = []
        if phase == PhaseType.PLANNING:
            lines.append(
                "The retrieved evidence mentions parameters that the design dependency graph "
                "links to others. Ensure the outline covers their causes and effects (topics "
                "only — do not add formulas):"
            )
            for cand, nb in findings:
                causes = ", ".join(self._pbds_label(c.node) for c in nb["sources"][:6]) or "none"
                effects = ", ".join(self._pbds_label(c.node) for c in nb["effects"][:6]) or "none"
                lines.append(f"- {cand.description}  |  caused by: {causes}  |  affects: {effects}")
        elif phase == PhaseType.DRAFTING:
            lines.append(
                "For accuracy while writing THIS section, the mentioned parameters relate as "
                "below. Use them to state causes/effects correctly; do NOT introduce new topics:"
            )
            for cand, nb in findings:
                for cn in (list(nb["sources"]) + list(nb["effects"]))[:6]:
                    if cn.direction == "sources":
                        rel = f"{cand.description} is computed from {self._pbds_label(cn.node)}"
                    else:
                        rel = f"{self._pbds_label(cn.node)} is computed from {cand.description}"
                    formula = cn.formulas[0] if cn.formulas else ""
                    lines.append(f"- {rel}" + (f"  (formula: {formula})" if formula else ""))
        else:  # RESEARCH
            lines.append(
                "The retrieved evidence discusses parameters whose design dependencies point to "
                "related parameters. Treat these as candidate next research targets so the report "
                "also covers their sources and effects:"
            )
            for cand, nb in findings:
                causes = ", ".join(self._pbds_label(c.node) for c in nb["sources"][:6]) or "none"
                effects = ", ".join(self._pbds_label(c.node) for c in nb["effects"][:6]) or "none"
                lines.append(f"- {cand.description} — depends on: {causes}; influences: {effects}")
        return "\n".join(lines)

    async def _pbds_block(self, documents) -> str:
        """Build the dependency-graph analysis block for the retrieved evidence, or "".

        No-op when the tool is inactive, the phase is not PBDS-relevant, or nothing
        matches. In PLANNING/RESEARCH the frontier set records every surfaced
        parameter so documents later retrieved FOR those parameters do not
        re-trigger the tool. DRAFTING annotates the current section only and does
        not touch the frontier. Never raises — the optional tool must not break
        retrieval.
        """
        matcher = self._pbds_matcher
        if matcher is None:
            return ""
        phase = self._pbds_phase()
        if phase is None:
            return ""
        from qmix_report_writer.handcrafted_graph.phases import PhaseType
        try:
            combined = "\n\n".join(str(d.get("content", "")) for d in documents)[:8000]
            if not combined.strip():
                return ""
            matched = await matcher.amatch(combined, self.llm, calling_agent="Researcher")
            apply_frontier = phase != PhaseType.DRAFTING
            findings = []
            for cand in matched:
                if apply_frontier and cand.node in self._pbds_surfaced_nodes:
                    continue
                nb = self._pbds_manager.neighborhood(cand.node, k=1)
                neighbors = list(nb["sources"]) + list(nb["effects"])
                if not neighbors:
                    continue
                if apply_frontier:
                    self._pbds_surfaced_nodes.add(cand.node)
                    self._pbds_surfaced_nodes.update(cn.node for cn in neighbors)
                findings.append((cand, nb))
                if len(findings) >= 4:
                    break
            return self._pbds_format_block(findings, phase) if findings else ""
        except Exception:
            return ""

    @staticmethod
    def _run_pbds_sync(coro):
        """Run a PBDS coroutine from the synchronous execute path."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)

        if self._is_revision_phase():
            da_output = self._get_da_output(spatial_info)
            if da_output is not None:
                # REVISION Round A (DataAnalyst→Researcher edge present).
                # Only retrieve evidence if DataAnalyst flagged a State Deficiency.
                if not _DEFICIENCY_RE.search(da_output):
                    if execution_trace:
                        execution_trace.trace[-1]["Researcher"]["response"] = "[HOLD]"
                    return "[HOLD]"
            else:
                # REVISION Round B (no incoming edge from DataAnalyst).
                # Forward prior evidence to DataAnalyst without a new RAG call.
                prior_outputs = self.last_memory.get("outputs") or []
                prior = str(prior_outputs[-1]).strip() if prior_outputs else ""
                is_hold = (
                    not prior
                    or prior.startswith("[HOLD]")
                    or prior.startswith("[RESEARCH_EXHAUSTED]")
                )
                result = "[HOLD]" if is_hold else prior
                if execution_trace:
                    execution_trace.trace[-1]["Researcher"]["response"] = result
                return result

        # Tool use
        system_prompt = self.prompt_set.get_description("RAG Tool") + self.prompt_set.get_constraint("RAG Tool")
        _, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["message_to"].append("Researcher")
            execution_trace.trace[-1]["Researcher"]["message_to"].append("RAG")
            execution_trace.trace[-1]["exec_order"].append("RAG")
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        raw_queries = self.llm.gen(message)
        queries = _parse_queries(raw_queries) if raw_queries and not raw_queries.strip().startswith("[") else []
        # PLANNING coverage scan must not be skipped. The Researcher runs before the
        # LeadArchitect with no directive yet, so the query-formulation LLM sometimes
        # returns nothing usable. Without a fallback the coverage scan yields
        # [RESEARCH_EXHAUSTED], the LA emits [AWAITING_COVERAGE_DATA], and PLANNING
        # produces no outline (triggering the generic drafting fallback). Use the
        # report subject as a broad topic-discovery query so the outline gets evidence.
        if not queries and self._is_planning_phase():
            subject = str((input or {}).get("task", "")).strip()
            if subject and not subject.startswith("["):
                queries = [subject]
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["prompt"] = queries
        if not queries:
            signal = "[RESEARCH_EXHAUSTED] RAG returned no documents for this query."
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["response"] = signal
                execution_trace.trace[-1]["Researcher"]["response"] = signal
            return signal
        documents = self.rag.query_docs_multi(queries, top_k=self._current_top_k())
        if documents and self._is_research_phase():
            documents = self._filter_reported_chunks(documents)
            if not documents:
                signal = self._DUPLICATE_RETRIEVAL_SIGNAL
                if execution_trace:
                    execution_trace.trace[-1]["RAG"]["response"] = signal
                    execution_trace.trace[-1]["Researcher"]["response"] = signal
                return signal
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"
            execution_trace.trace[-1]["RAG"]["sources"] = [
                {
                    "id": d["id"],
                    "source": d["source"],
                    "page": d.get("page", "N/A"),
                    "vector_distance": d.get("distance"),
                    "bm25_score": d.get("bm25_score"),
                    "rrf_score": d.get("rrf_score"),
                    "in_vector": d.get("in_vector", False),
                    "in_bm25": d.get("in_bm25", False),
                    "nomic_score": d.get("nomic_score"),
                    "reranker_score": d.get("reranker_score"),
                }
                for d in documents
            ]

        if not documents:
            signal = "[RESEARCH_EXHAUSTED] RAG returned no documents for this query."
            if execution_trace:
                execution_trace.trace[-1]["Researcher"]["response"] = signal
            return signal

        for i, document in enumerate(documents):
            citation = f"<source> {document['source']} </source>\n<content>\n{document['content']}\n</content>"
            spatial_info[f"Data_{i}"] = {"role": "RAG Tool", "output": citation}
            SourceBuffer.instance().add(document)

        # Base execution
        pbds_block = (
            self._run_pbds_sync(self._pbds_block(documents))
            if self._pbds_matcher is not None
            else ""
        )
        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info, pbds_block=pbds_block, **kwargs
        )
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, calling_agent="Researcher")
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["response"] = response
        if self._is_planning_phase():
            self._persist_deficiencies(response)
        return response

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)

        if self._is_revision_phase():
            da_output = self._get_da_output(spatial_info)
            if da_output is not None:
                # REVISION Round A (DataAnalyst→Researcher edge present).
                # Only retrieve evidence if DataAnalyst flagged a State Deficiency.
                if not _DEFICIENCY_RE.search(da_output):
                    if execution_trace:
                        execution_trace.trace[-1]["Researcher"]["response"] = "[HOLD]"
                    return "[HOLD]"
            else:
                # REVISION Round B (no incoming edge from DataAnalyst).
                # Forward prior evidence to DataAnalyst without a new RAG call.
                prior_outputs = self.last_memory.get("outputs") or []
                prior = str(prior_outputs[-1]).strip() if prior_outputs else ""
                is_hold = (
                    not prior
                    or prior.startswith("[HOLD]")
                    or prior.startswith("[RESEARCH_EXHAUSTED]")
                )
                result = "[HOLD]" if is_hold else prior
                if execution_trace:
                    execution_trace.trace[-1]["Researcher"]["response"] = result
                return result

        # Tool use
        system_prompt = self.prompt_set.get_description("RAG Tool") + self.prompt_set.get_constraint("RAG Tool")
        _, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["message_to"].append("Researcher")
            execution_trace.trace[-1]["Researcher"]["message_to"].append("RAG")
            execution_trace.trace[-1]["exec_order"].append("RAG")
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        raw_queries = await self.llm.agen(message)
        queries = _parse_queries(raw_queries) if raw_queries and not raw_queries.strip().startswith("[") else []
        # PLANNING coverage scan must not be skipped. The Researcher runs before the
        # LeadArchitect with no directive yet, so the query-formulation LLM sometimes
        # returns nothing usable. Without a fallback the coverage scan yields
        # [RESEARCH_EXHAUSTED], the LA emits [AWAITING_COVERAGE_DATA], and PLANNING
        # produces no outline (triggering the generic drafting fallback). Use the
        # report subject as a broad topic-discovery query so the outline gets evidence.
        if not queries and self._is_planning_phase():
            subject = str((input or {}).get("task", "")).strip()
            if subject and not subject.startswith("["):
                queries = [subject]
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["prompt"] = queries
        if not queries:
            signal = "[RESEARCH_EXHAUSTED] RAG returned no documents for this query."
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["response"] = signal
                execution_trace.trace[-1]["Researcher"]["response"] = signal
            return signal
        documents = self.rag.query_docs_multi(queries, top_k=self._current_top_k())
        if documents and self._is_research_phase():
            documents = self._filter_reported_chunks(documents)
            if not documents:
                signal = self._DUPLICATE_RETRIEVAL_SIGNAL
                if execution_trace:
                    execution_trace.trace[-1]["RAG"]["response"] = signal
                    execution_trace.trace[-1]["Researcher"]["response"] = signal
                return signal
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"
            execution_trace.trace[-1]["RAG"]["sources"] = [
                {
                    "id": d["id"],
                    "source": d["source"],
                    "page": d.get("page", "N/A"),
                    "vector_distance": d.get("distance"),
                    "bm25_score": d.get("bm25_score"),
                    "rrf_score": d.get("rrf_score"),
                    "in_vector": d.get("in_vector", False),
                    "in_bm25": d.get("in_bm25", False),
                    "nomic_score": d.get("nomic_score"),
                    "reranker_score": d.get("reranker_score"),
                }
                for d in documents
            ]

        if not documents:
            signal = "[RESEARCH_EXHAUSTED] RAG returned no documents for this query."
            if execution_trace:
                execution_trace.trace[-1]["Researcher"]["response"] = signal
            return signal

        for i, document in enumerate(documents):
            citation = f"<source> {document['source']} </source>\n<content>\n{document['content']}\n</content>"
            spatial_info[f"Data_{i}"] = {"role": "RAG Tool", "output": citation}
            SourceBuffer.instance().add(document)

        # Base execution
        pbds_block = await self._pbds_block(documents) if self._pbds_matcher is not None else ""
        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info, pbds_block=pbds_block, **kwargs
        )
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, calling_agent="Researcher")
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["response"] = response
        if self._is_planning_phase():
            self._persist_deficiencies(response)
        return response

if __name__ == "__main__":
    import asyncio
    import shortuuid
    input_arg = {"task": "What is the purity rate of Graphene?"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = Researcher(llm_name="tinyllama")

    dummy_docs = [
        "The boiling point of Liquid X-42 is 156.4 degrees Celsius under standard pressure.",
        "Graphene synthesis via chemical vapor deposition shows a 98% purity rate when using copper substrates.",
        "The proprietary 'Alpha-Protocol' requires a mixture of 10% Argon and 90% Nitrogen for stable plasma.",
        "Clinical trials for Compound-9 revealed a significant reduction in neural inflammation within 48 hours."
    ]

    metadatas = [
        {"source_name": "Lab_Results_2026.pdf"},
        {"source_name": "Material_Science_Journal.docx"},
        {"source_name": "Engineering_Manual_v2.txt"},
        {"source_name": "Medical_Report_Draft.pdf"}
    ]

    ids = [f"id_{shortuuid.uuid()}" for _ in range(len(dummy_docs))]
    col.rag.add_documents(dummy_docs, metadatas, ids)

    asyncio.run(col.async_execute(input_arg))
    print(SourceBuffer.instance().sources)
    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
