from typing import List, Any, Dict

from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from prompt.prompt_set_registry import PromptSetRegistry
from tools.coding.python_executor import execute_code_get_return


@AgentRegistry.register("CodeWriter")
class CodeWriting(Node):
    def __init__(self, id=None, role=None, domain="", llm_name=""):
        super().__init__(id, "CodeWriting", domain, llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        self.internal_tests = []

    def _extract_tests(self, prompt_text: str) -> list:
        lines = (line.strip() for line in prompt_text.split("\n") if line.strip())
        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith(">>>"):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")
        return results

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""

        for agent_id, info in spatial_info.items():
            output = info["output"]
            spatial_str += f"Agent {agent_id} ({info['role']}): {output}\n\n"

        for agent_id, info in temporal_info.items():
            temporal_str += f"Agent {agent_id} ({info['role']}) previous: {info['output']}\n\n"

        user_prompt = f"The task is:\n\n{raw_inputs['task']}\n"
        if spatial_str:
            user_prompt += f"\nOther agents' outputs:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious round outputs:\n{temporal_str}"

        return system_prompt, user_prompt

    def _execute(self, input_data, spatial_info, temporal_info, **kwargs):
        self.internal_tests = self._extract_tests(input_data.get("task", ""))
        system_prompt, user_prompt = self._process_inputs(input_data, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.gen(message)

    async def _async_execute(self, input_data, spatial_info, temporal_info, **kwargs):
        self.internal_tests = self._extract_tests(input_data.get("task", ""))
        system_prompt, user_prompt = self._process_inputs(input_data, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return await self.llm.agen(message)
