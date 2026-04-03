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
