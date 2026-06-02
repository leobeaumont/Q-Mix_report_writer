import re

from graph.node import Node
from agents.agent_registry import AgentRegistry
from tools.rag import RAGManager
from utils.config import get_llm
from utils.globals import ReportState, SourceBuffer
from prompt.prompt_set_registry import PromptSetRegistry

_DEFICIENCY_RE = re.compile(r"State Deficiency", re.IGNORECASE)


@AgentRegistry.register("Researcher")
class Researcher(Node):
    """Main source of information of the team, here to find documents and citations."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Researcher", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Researcher"
        self.report = ReportState.instance()
        self.rag = RAGManager()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Current report state", self.report.progress,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _is_revision_phase(self) -> bool:
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.REVISION
        except Exception:
            return False

    def _is_planning_phase(self) -> bool:
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.PLANNING
        except Exception:
            return False

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
            execution_trace.trace[-1]["RAG"]["prompt"] = system_prompt + user_prompt
            execution_trace.trace[-1]["RAG"]["message_to"].append("Researcher")
            execution_trace.trace[-1]["Researcher"]["message_to"].append("RAG")
            execution_trace.trace[-1]["exec_order"].append("RAG")
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        query = self.llm.gen(message)
        documents = self.rag.query_docs(query)
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"

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
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
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
        query = await self.llm.agen(message)
        documents = self.rag.query_docs(query)
        if execution_trace:
            execution_trace.trace[-1]["RAG"]["prompt"] = query
            execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"

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
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
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
