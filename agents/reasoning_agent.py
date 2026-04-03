from typing import Dict, Any

from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("ReasoningAgent")
class ReasoningAgent(Node):
    """General-purpose reasoning agent for MMLU, GAIA, Frontier Science, etc."""

    def __init__(self, id=None, role=None, domain="", llm_name=""):
        super().__init__(id, "ReasoningAgent", domain, llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = role or "Reasoning Expert"
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = (
            "You are a world-class reasoning expert. Think step by step, "
            "consider multiple perspectives, and provide a well-reasoned answer. "
            "Be concise but thorough."
        )

        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}):\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} ({info['role']}) previously:\n{info['output']}\n\n"

        user_prompt = f"Task: {raw_inputs['task']}\n"
        if spatial_str:
            user_prompt += f"\nOther agents' current responses:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious round:\n{temporal_str}"
        user_prompt += "\nThink step by step and provide your answer."

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.gen(message)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return await self.llm.agen(message)
