from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Reviewer")
class Reviewer(Node):
    """Text reviewer of the team, here to ensure the quality of the production."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Reviewer", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Reviewer"
        self.report = ReportState.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        # Reviewer needs the report text for auditing, but capped to avoid
        # context overflow on small models. We take the tail (most recent content).
        _MAX_CHARS = 6000
        full = self.report.content or self.report.progress
        if len(full) > _MAX_CHARS:
            report_content = f"[...truncated, showing last {_MAX_CHARS} chars...]\n" + full[-_MAX_CHARS:]
        else:
            report_content = full
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Full Report Content", report_content,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, calling_agent="Reviewer")
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["response"] = response
        return response

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, calling_agent="Reviewer")
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["response"] = response
        return response

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = Reviewer(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
