from typing import List, Any, Dict

from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("MathSolver")
class MathSolver(Node):
    def __init__(self, id=None, role=None, domain="", llm_name=""):
        super().__init__(id, "MathSolver", domain, llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)

        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}) answered:\n\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} ({info['role']}) previously answered:\n\n{info['output']}\n\n"

        if spatial_str:
            user_prompt += f"\nOther agents' current answers:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious round answers:\n{temporal_str}"

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.gen(message)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message)
        return response
