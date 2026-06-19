"""Q-Mix report writer — agentic RAG-based report generation.

Public entry point for embedding the handcrafted pipeline in a host project:

    from qmix_report_writer.handcrafted_graph import run_handcrafted
    from qmix_report_writer.utils.config import configure

    configure(config_path="my_host_config.yaml")  # optional override
    answers, tokens = await run_handcrafted(task="...")
"""

from qmix_report_writer.handcrafted_graph import run_handcrafted

__all__ = ["run_handcrafted"]
