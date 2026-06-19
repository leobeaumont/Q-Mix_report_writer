"""Q-Mix report writer — agentic RAG-based report generation.

Public entry point for embedding the handcrafted pipeline in a host project:

    from qmix_report_writer.handcrafted_graph import run_handcrafted
    from qmix_report_writer.utils.config import configure

    configure(config_path="my_host_config.yaml")  # optional override
    answers, tokens = await run_handcrafted(task="...")
"""

__all__ = ["run_handcrafted"]


def __getattr__(name):
    # Lazy public export: importing this package (or a light submodule such as
    # utils.config) must not eagerly drag in the heavy pipeline dependencies
    # (torch, chromadb, ...). run_handcrafted is resolved on first access.
    if name == "run_handcrafted":
        from qmix_report_writer.handcrafted_graph import run_handcrafted
        return run_handcrafted
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
