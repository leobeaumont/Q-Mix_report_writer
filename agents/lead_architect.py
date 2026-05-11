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

        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"#### Message from {info['role']}:\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"#### Previous output from {info['role']}:\n{info['output']}\n\n"

        user_prompt = f"\n\n### Report Subject:\n{raw_inputs['task']}\n"

        user_prompt += f"\n### Current report state:\n{self.report.progress}\n"

        user_prompt += f"\n### Current Team Objective:\n{self.report.task}\n"

        if spatial_str:
            user_prompt += f"\n### Received messages:\n\n{spatial_str}"
        if temporal_str:
            user_prompt += f"### Your previous output:\n\n{temporal_str}"
        user_prompt += "### [WRITE OUTPUT HERE]"

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        schema = self.prompt_set.get_schema(self.role)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, response_schema=schema)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        response = safe_json_parse(response)
        self.report.task = response.get("current_task", self.report.task)
        return response.get("strategy", "")

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        schema = self.prompt_set.get_schema(self.role)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, response_schema=schema)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        response = safe_json_parse(response)
        self.report.task = response.get("current_task", self.report.task)
        return response.get("strategy", "")

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = LeadArchitect(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
