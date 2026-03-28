from typing import Dict, Any, List

from graph.node import Node
from agents.agent_registry import AgentRegistry
from llm.llm_registry import LLMRegistry
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("FinalDirect")
class FinalDirect(Node):
    """Passes through the last agent's output directly."""

    def __init__(self, id=None, domain="", llm_name=""):
        super().__init__(id, "FinalDirect")
        self.role = "Decision Maker"

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        return None

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        for info in spatial_info.values():
            return info["output"]
        return "No answer"

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        for info in spatial_info.values():
            return info["output"]
        return "No answer"


@AgentRegistry.register("FinalRefer")
class FinalRefer(Node):
    """LLM-based decision node that synthesizes all agents' outputs."""

    def __init__(self, id=None, domain="", llm_name=""):
        super().__init__(id, "FinalRefer", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Final Decision Maker"

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        decision_role = self.prompt_set.get_decision_role().strip()
        decision_constraint = self.prompt_set.get_decision_constraint().strip()
        system_prompt = f"{decision_role}\n\n{decision_constraint}"

        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}):\n{info['output']}\n\n"

        user_prompt = f"Task: {raw_inputs['task']}\n\nAgents' responses:\n{spatial_str}"
        user_prompt += "\nSynthesize the best final answer."
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.gen(message)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return await self.llm.agen(message)


@AgentRegistry.register("FinalMajorVote")
class FinalMajorVote(Node):
    """Majority-vote decision node."""

    def __init__(self, id=None, domain="", llm_name=""):
        super().__init__(id, "FinalMajorVote")
        self.role = "Majority Voter"

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        return None

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        return self._vote(spatial_info)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return self._vote(spatial_info)

    def _vote(self, spatial_info):
        from utils.utils import extract_number, extract_choice

        counts = {}
        best = ""
        best_count = 0
        for info in spatial_info.values():
            output = info["output"].strip()
            num = extract_number(output)
            key = str(num) if num is not None else output[-200:]

            counts[key] = counts.get(key, 0) + 1
            if counts[key] > best_count:
                best = output
                best_count = counts[key]
        return best
