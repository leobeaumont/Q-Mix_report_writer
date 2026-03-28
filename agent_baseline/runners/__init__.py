"""Framework runner factory – lazy imports to avoid pulling every framework."""

from .base_runner import BaseRunner


def get_runner(name: str, *, base_url: str, api_key: str, model: str) -> BaseRunner:
    if name == "agent-framework":
        from .agent_framework_runner import AgentFrameworkRunner
        return AgentFrameworkRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "autogen":
        from .autogen_runner import AutoGenRunner
        return AutoGenRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "langgraph":
        from .langgraph_runner import LangGraphRunner
        return LangGraphRunner(base_url=base_url, api_key=api_key, model=model)
    elif name == "lobster":
        from .lobster_runner import LobsterRunner
        return LobsterRunner(base_url=base_url, api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown framework: {name}. "
            f"Available: agent-framework, autogen, langgraph, lobster"
        )
