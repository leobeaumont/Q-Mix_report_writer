from graph.node import Node
from agents.agent_registry import AgentRegistry
from tools.rag import RAGManager
from utils.config import get_llm
from utils.globals import ReportState, SourceBuffer
from prompt.prompt_set_registry import PromptSetRegistry


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

        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}):\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} ({info['role']}) previously:\n{info['output']}\n\n"

        user_prompt = f"Task: {raw_inputs['task']}\n"

        user_prompt += f"\nCurrent report state: {ReportState.instance().progress}\n"

        if spatial_str:
            user_prompt += f"\nOther agent's current responses:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious round:\n{temporal_str}"
        user_prompt += "\nThink step by step and provide your answer."

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)

        # Tool use
        action = kwargs.get("action", None)
        if action == 8:  # when using execute_verify
            system_prompt = self.prompt_set.get_description("RAG Tool") + self.prompt_set.get_constraint("RAG Tool")
            _, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["prompt"] = system_prompt + user_prompt
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            query = self.llm.gen(message)
            documents = self.rag.query_docs(query)
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"
            for i, document in enumerate(documents):
                citation = f"<source> {document["source"]} </source>\n<content>\n{document["content"]}\n</content>"
                spatial_info[f"Data_{i}"] = {"role": "RAG Tool", "output": citation}
                SourceBuffer.instance().add(document)
        
        # Base execution
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message)
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["response"] = response
        return response

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)

        # Tool use
        action = kwargs.get("action", None)
        if action == 8:  # when using execute_verify
            system_prompt = self.prompt_set.get_description("RAG Tool") + self.prompt_set.get_constraint("RAG Tool")
            _, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["prompt"] = system_prompt + user_prompt
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            query = await self.llm.agen(message)
            documents = self.rag.query_docs(query)
            if execution_trace:
                execution_trace.trace[-1]["RAG"]["response"] = f"{documents}"
            for i, document in enumerate(documents):
                citation = f"<source> {document["source"]} </source>\n<content>\n{document["content"]}\n</content>"
                spatial_info[f"Data_{i}"] = {"role": "RAG Tool", "output": citation}
                SourceBuffer.instance().add(document)
        
        # Base execution
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message)
        if execution_trace:
            execution_trace.trace[-1]["Researcher"]["response"] = response
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
