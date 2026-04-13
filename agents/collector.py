from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Collector")
class Collector(Node):
    """Collector agent meant to collect report text as it is written."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Collector", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("collector")
        self.role = role or "Collector"
        self.report = ReportState.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs): 
        system_prompt = self.prompt_set.get_role() 
        system_prompt += self.prompt_set.get_description(self.role) 
        system_prompt += self.prompt_set.get_constraint(self.role)

        previous_paragraph = self.report.get_last()
        if spatial_info:
            _, current_text = spatial_info.popitem()
        else:
            current_text = "[NO CURRENT TEXT]"

        user_prompt = f"""
# INPUT DATA
<previous_paragraph>
{previous_paragraph}
</previous_paragraph>

<current_text>
{current_text}
</current_text>

# INSTRUCTION
Based on the rules in the system prompt, output the cleaned and transitioned version of the <current_text> below.
"""
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = self.llm.agen(message)

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = self.llm.agen(message)

        self.report.append(response1, response2)
        return response1

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = await self.llm.agen(message)

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = await self.llm.agen(message)

        self.report.append(response1, response2)
        return response1
    
    def _progress_prompt(self, previous_progress: str, addition: str) -> str:
        task = self.prompt_set.get_summarize()

        context = f"""
# INPUT DATA
<report state>
{previous_progress}
</report state>

<addition>
{addition}
</addition>

[WRITE NEW SUMMARY HERE]
"""
        return task, context
    
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

