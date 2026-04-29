from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState, SourceBuffer
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Collector")
class Collector(Node):
    """Collector agent meant to collect report text as it is written."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Collector", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Collector"
        self.report = ReportState.instance()
        self.source_buffer = SourceBuffer.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs): 
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)

        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"#### Message from {info['role']}:\n{info['output']}\n\n"

        user_prompt = f"\n\n### Task:\n{raw_inputs['task']}\n"

        user_prompt += f"\n### Previous Text Production:\n{ReportState.instance().get_last()}\n"

        if spatial_str:
            user_prompt += f"\n### Received messages:\n\n{spatial_str}"
        user_prompt += "### [WRITE OUTPUT HERE]"

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        execution_trace = kwargs.get("execution_trace", None)
        
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = self.llm.gen(message)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["response"] = response1

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = self.llm.gen(message)

        new_sources = self.source_buffer.flush()
        self.report.append(response1, response2, new_sources)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["report_state"] = self.report.content
        return response1

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        execution_trace = kwargs.get("execution_trace", None)
        
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = await self.llm.agen(message)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["response"] = response1

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = await self.llm.agen(message)

        new_sources = self.source_buffer.flush()
        self.report.append(response1, response2, new_sources)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["report_state"] = self.report.content
        return response1
    
    def _progress_prompt(self, previous_progress: str, addition: str) -> str:
        system_prompt = self.prompt_set.get_description("Summarizer")

        user_prompt = f"\n\n### Previous summary:\n{previous_progress}\n"
        user_prompt += f"\n### New addition:\n{addition}\n"
        user_prompt += "### [WRITE NEW SUMMARY HERE]"

        return system_prompt, user_prompt
    
if __name__ == "__main__":
    spatial = {"key": "This is the current text"}
    col = Collector(llm_name="tinyllama")

    system, user = col._process_inputs([], spatial, {})

    print(system)
    print("=" * 60)
    print(user)

    old = "Old report state."
    new = "New addition."

    system, user = col._progress_prompt(old, new)
    print("\n\n")
    print(system)
    print("=" * 60)
    print(user)
