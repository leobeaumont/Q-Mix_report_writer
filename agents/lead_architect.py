from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
from utils.utils import safe_json_parse
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("LeadArchitect")
class LeadArchitect(Node):
    """Lead of the redaction team, here to plan and structure the writing of the report."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "LeadArchitect", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Lead Architect"
        self.report = ReportState.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        # Expose the structured section list alongside the prose summary so the
        # model can reliably detect which topics have already been written.
        section_list = self.report.list_sections()
        report_context = f"Sections written so far:\n{section_list}\n\nProgress summary:\n{self.report.progress}"
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Current report state", report_context,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        schema = self.prompt_set.get_schema(self.role)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, response_schema=schema, calling_agent="LeadArchitect")
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        response = safe_json_parse(response)
        current_task = response.get("current_task", "")
        self.report.task = current_task or self.report.task
        strategy = response.get("strategy", "")
        if current_task:
            return f"{strategy}\n\n**Assigned task:** {current_task}"
        return strategy

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        schema = self.prompt_set.get_schema(self.role)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, response_schema=schema, calling_agent="LeadArchitect")
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        response = safe_json_parse(response)
        current_task = response.get("current_task", "")
        self.report.task = current_task or self.report.task
        strategy = response.get("strategy", "")
        if current_task:
            return f"{strategy}\n\n**Assigned task:** {current_task}"
        return strategy

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = LeadArchitect(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
