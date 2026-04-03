import yaml
from .const import PROJECT_ROOT

_config = None


def get_config() -> dict:
    """Load and cache the default.yaml config. Returns the full config as a dict."""
    global _config
    if _config is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_llm_config() -> dict:
    """Convenience accessor for the llm section of the config."""
    return get_config().get("llm", {})


def get_llm(llm_name: str):
    """
    Instantiate an LLM from the registry using settings from default.yaml.
    
    This is the single correct way to get an LLM instance anywhere in the
    project — it ensures temperature and max_tokens from the config are
    always respected rather than silently falling back to hardcoded defaults.

    Usage (replaces LLMRegistry.get(llm_name, model_name=llm_name)):
        from utils.config import get_llm
        self.llm = get_llm(llm_name)
    """
    from llm.llm_registry import LLMRegistry
    llm_cfg = get_llm_config()
    return LLMRegistry.get(
        llm_name,
        model_name=llm_name,
        temperature=llm_cfg.get("temperature", 0.2),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )
