import yaml
from .const import PROJECT_ROOT
from typing import Optional

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


def get_llm(llm_name: Optional[str] = None):
    """
    Instantiate an LLM from the provider config in default.yaml.

    To add or remove models, edit llm.providers in default.yaml — no code changes needed.
    To use a specific model, pass its name here or set llm.default_model in default.yaml.
    """
    from llm.ollama_chat import OllamaChat

    llm_cfg = get_llm_config()
    if llm_name is None:
        llm_name = llm_cfg.get("default_model", "qwen3:8b")

    providers = llm_cfg.get("providers", {})
    for provider_name, provider_cfg in providers.items():
        if llm_name in provider_cfg.get("models", []):
            if provider_name == "ollama":
                return OllamaChat(
                    model_name=llm_name,
                    temperature=llm_cfg.get("temperature", 0.7),
                    top_p=llm_cfg.get("top_p", 0.8),
                    top_k=llm_cfg.get("top_k", 20),
                    min_p=llm_cfg.get("min_p", 0.0),
                    max_tokens=llm_cfg.get("max_tokens", 4096),
                )
            raise ValueError(f"Provider '{provider_name}' is not supported yet.")

    raise ValueError(
        f"Model '{llm_name}' was not found under any provider in llm.providers. "
        f"Add it to the relevant provider's models list in configs/default.yaml."
    )
