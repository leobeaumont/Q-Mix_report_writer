from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Researcher")
class Researcher(Node):
    """Main source of information of the team, here to find documents and citations."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Researcher", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Researcher"
        self.report = ReportState.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)

        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} ({info['role']}):\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} ({info['role']}) previously:\n{info['output']}\n\n"

        user_prompt = f"Task: {raw_inputs['task']}\n"

        user_prompt += f"\nCurrent report state: {ReportState.instance().progress}\n"

        if spatial_str:
            user_prompt += f"\nOther agent's current responses:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious round:\n{temporal_str}"
        user_prompt += "\nThink step by step and provide your answer."

        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        action = kwargs.get("action", None)
        if action == 8:  # when using execute_verify
            # Implement tool use here
            pass
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.gen(message)

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        action = kwargs.get("action", None)
        if action == 8:  # when using execute_verify
            # Implement tool use here
            pass
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return await self.llm.agen(message)

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = Researcher(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
