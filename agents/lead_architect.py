import re

from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
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
        # In PLANNING, surface any topics the Researcher confirmed are absent from
        # the knowledge base so the LLM cannot accidentally plan sections for them.
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            if (
                PhaseState.instance().current_phase == PhaseType.PLANNING
                and self.report.deficient_topics
            ):
                absent_block = "\n\nTopics absent from knowledge base — do NOT plan sections for these:\n"
                absent_block += "\n".join(f"- {t}" for t in self.report.deficient_topics)
                report_context += absent_block
        except Exception:
            pass
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Current report state", report_context,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _parse_response(self, response: str):
        if "[DRAFTING_COMPLETE]" in response:
            strategy = response.replace("[DRAFTING_COMPLETE]", "").strip()
            return "[DRAFTING_COMPLETE]", strategy
        if "[REVISION_COMPLETE]" in response:
            strategy = response.replace("[REVISION_COMPLETE]", "").strip()
            return "[REVISION_COMPLETE]", strategy
        match = re.search(r"<task>(.*?)</task>", response, re.DOTALL)
        current_task = match.group(1).strip() if match else "Continue developing the report based on the current plan."
        strategy = re.sub(r"<task>.*?</task>", "", response, flags=re.DOTALL).strip()
        return current_task, strategy

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, calling_agent="LeadArchitect")
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        current_task, strategy = self._parse_response(response)
        self.report.task = current_task
        return f"{strategy}\n\n**Assigned task:** {current_task}"

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, calling_agent="LeadArchitect")
        if execution_trace:
            execution_trace.trace[-1]["LeadArchitect"]["response"] = response
        current_task, strategy = self._parse_response(response)
        self.report.task = current_task
        return f"{strategy}\n\n**Assigned task:** {current_task}"

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = LeadArchitect(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
