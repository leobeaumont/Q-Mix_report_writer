import os
import yaml
from pathlib import Path
from .const import PACKAGE_ROOT
from typing import Optional

_config = None

# Environment variables a host can set instead of calling configure() in code.
ENV_CONFIG_PATH = "QMIX_REPORT_CONFIG"        # path to a YAML override file
ENV_DATA_ROOT = "QMIX_REPORT_DATA_ROOT"       # base dir for the project's data (e.g. the DB)
ENV_OUTPUT_ROOT = "QMIX_REPORT_OUTPUT_ROOT"   # base dir for PRODUCED files (reports, traces)
ENV_PBDS_WORKBOOK = "QMIX_REPORT_PBDS_WORKBOOK"  # direct path to the PBDS parameter workbook


def _load_defaults() -> dict:
    """Load the bundled default.yaml that ships inside the package."""
    config_path = PACKAGE_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` in place (override wins)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def configure(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    """Configure the package from a host project.

    Call once at host startup, before the first get_config()/run. Resolution order
    (each layer overrides the previous):
      1. bundled default.yaml
      2. YAML file at ``config_path`` (or the ``QMIX_REPORT_CONFIG`` env var)
      3. the ``overrides`` dict

    Returns the merged config and caches it for subsequent get_config() calls.
    """
    global _config
    cfg = _load_defaults()
    path = config_path or os.environ.get(ENV_CONFIG_PATH)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_merge(cfg, overrides)
    _config = cfg
    return _config


def get_config() -> dict:
    """Return the active config, building it from defaults (+ env overrides) on first use."""
    global _config
    if _config is None:
        configure()
    return _config


def get_llm_config() -> dict:
    """Convenience accessor for the llm section of the config."""
    return get_config().get("llm", {})


def get_rag_config() -> dict:
    """Convenience accessor for the rag section of the config."""
    return get_config().get("rag", {})


# ---------------------------------------------------------------------------
# Runtime data paths
# ---------------------------------------------------------------------------
# Bundled resources (configs/, assets/) live inside the package and resolve via
# PACKAGE_ROOT. Two configurable roots govern everything else, kept deliberately
# separate so a host can relocate generated artifacts without disturbing the DB:
#
#   data root   — base for *used* resources, notably the vector DB. Defaults to
#                 the current working directory (= the project root when run from
#                 there), so standalone use keeps finding ./chroma_data.
#   output root — base for files the pipeline *produces* (reports, traces).
#                 Defaults to the data root, so standalone keeps writing at the
#                 project root. A host typically points this at a dedicated folder
#                 (e.g. "qmix_report_writer_data") to keep generated artifacts
#                 grouped. The DB is NOT placed here.


def _resolve_under(base: Path, value, default: str) -> Path:
    """Resolve a configured value to an absolute Path. Absolute values are used
    as-is; relative values (and the default) are taken relative to ``base``."""
    p = Path(value if value is not None else default).expanduser()
    return p if p.is_absolute() else base / p


def get_data_root() -> Path:
    """Base directory for the project's *used* data (e.g. the vector DB).

    Resolution: paths.data_root → QMIX_REPORT_DATA_ROOT → current working
    directory (the project root when run from there — standalone is unaffected).
    """
    paths = get_config().get("paths", {}) or {}
    root = paths.get("data_root") or os.environ.get(ENV_DATA_ROOT)
    return Path(root).expanduser() if root else Path.cwd()


def get_output_root() -> Path:
    """Base directory for files the pipeline *produces* (reports, traces).

    Resolution: paths.output_root → QMIX_REPORT_OUTPUT_ROOT → the data root.
    Relative values resolve under the data root, so a host can set
    ``output_root: qmix_report_writer_data`` to group all generated files in one
    folder at its root, while leaving the DB where it is.
    """
    paths = get_config().get("paths", {}) or {}
    value = paths.get("output_root") or os.environ.get(ENV_OUTPUT_ROOT)
    if not value:
        return get_data_root()
    p = Path(value).expanduser()
    return p if p.is_absolute() else get_data_root() / p


def get_chroma_path() -> str:
    """Filesystem path for the ChromaDB persistent client (a *used* resource).

    Resolved against the data root, independently of the output root, so the DB
    never lands inside the produced-artifacts folder. Default: <data_root>/chroma_data.
    """
    paths = get_config().get("paths", {}) or {}
    return str(_resolve_under(get_data_root(), paths.get("chroma_path"), "chroma_data"))


def get_tools_dir() -> Path:
    """Cache directory for the auto-downloaded Tectonic (LaTeX) binary.

    Machine-local downloaded state — never inside the installed package, which may
    be read-only. Resolved against the data root. Default: <data_root>/.tools. A
    host with Tectonic on PATH never triggers a download, so this only matters as
    a writable fallback.
    """
    paths = get_config().get("paths", {}) or {}
    return _resolve_under(get_data_root(), paths.get("tools_dir"), ".tools")


def get_output_dir() -> Path:
    """Directory holding one sub-folder per report run (a produced artifact)."""
    paths = get_config().get("paths", {}) or {}
    return _resolve_under(get_output_root(), paths.get("output_dir"), "output")


def get_trace_path(filename: str = "handcrafted_trace.json") -> Path:
    """Path for a saved execution trace (a produced artifact)."""
    paths = get_config().get("paths", {}) or {}
    return _resolve_under(get_output_root(), paths.get("trace_file"), filename)


def get_pbds_workbook_path() -> Path:
    """Configured path to the PBDS parameter workbook (.xlsx) — may not exist.

    The workbook is a *used* input resource (like the vector DB), so it resolves
    against the data root. Resolution (each overrides the previous):
      1. pbds.workbook_path in the config — set in default.yaml, a QMIX_REPORT_CONFIG
         override file, or a configure(overrides=...) dict passed by a host.
      2. the QMIX_REPORT_PBDS_WORKBOOK environment variable.
      3. the default filename 'pbds_parameters.xlsx'.
    Relative values resolve under the data root; absolute values are used as-is.
    This only resolves the path — see get_active_pbds_workbook() for the existence
    gate that decides whether the tool activates.
    """
    pbds = get_config().get("pbds", {}) or {}
    value = pbds.get("workbook_path") or os.environ.get(ENV_PBDS_WORKBOOK)
    return _resolve_under(get_data_root(), value, "pbds_parameters.xlsx")


def get_active_pbds_workbook() -> Optional[Path]:
    """The PBDS workbook to use, or None when no readable file exists there.

    Returns the resolved path only if a file is present, so callers can leave the
    pipeline unchanged (the tool silently stays off) when the workbook is absent.
    """
    path = get_pbds_workbook_path()
    return path if path.is_file() else None


def get_llm(llm_name: Optional[str] = None):
    """
    Instantiate an LLM from the provider config in default.yaml.

    To add or remove models, edit llm.providers in default.yaml — no code changes needed.
    To use a specific model, pass its name here or set llm.default_model in default.yaml.
    """
    from qmix_report_writer.llm.ollama_chat import OllamaChat

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
