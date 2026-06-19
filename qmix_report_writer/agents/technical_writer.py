from qmix_report_writer.graph.node import Node
from qmix_report_writer.agents.agent_registry import AgentRegistry
from qmix_report_writer.utils.config import get_llm
from qmix_report_writer.utils.globals import ReportState
from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("TechnicalWriter")
class TechnicalWriter(Node):
    """Redactor of the team, here to build the final text from drafts and ideas."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "TechnicalWriter", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Technical Writer"
        self.report = ReportState.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Current report state", self.report.progress,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["TechnicalWriter"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, calling_agent="TechnicalWriter")
        if execution_trace:
            execution_trace.trace[-1]["TechnicalWriter"]["response"] = response
        return response

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["TechnicalWriter"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, calling_agent="TechnicalWriter")
        if execution_trace:
            execution_trace.trace[-1]["TechnicalWriter"]["response"] = response
        return response

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = TechnicalWriter(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
