"""
Execution-trace round helpers, shared by graph implementations.

Extracted from HandcraftedGraph (upgrade-plan Stage 2.3) so the round-slot
schema lives in one place — it is the schema StandaloneVisualizer
(utils/visualization.py) renders. The per-agent "action" field is None for
handcrafted runs; a QMIX controller populates it (upgrade-plan Stage 4.2).
"""

from typing import Any, Dict

from qmix_report_writer.utils.globals import ReportState


def init_trace_round(execution_trace, agent_names, collector_id) -> None:
    """Append a fresh round slot to the execution trace.

    Schema (identical for handcrafted and QMIX-controlled runs):
      - One entry per agent name (action=None unless a controller supplies one)
      - "RAG" entry (populated later by the Researcher agent)
      - "PBDS" entry (populated by the Researcher's PBDS tool when active)
      - "Collector" entry with report_state snapshot
      - "exec_order" list (populated as agents execute)
    """
    round_data: Dict[str, Any] = {
        name: {
            "action": None,
            "message_to": [],
            "prompt": None,
            "response": None,
            "time": None,
            "completion_tokens": None,
        }
        for name in agent_names
    }
    round_data["RAG"] = {"action": None, "message_to": [], "prompt": None, "response": None, "sources": []}
    round_data["PBDS"] = {"action": None, "message_to": [], "prompt": None, "response": None}
    if collector_id is not None:
        round_data["Collector"]["report_state"] = ReportState.instance().content
    round_data["exec_order"] = []
    execution_trace.trace.append(round_data)


def trace_spatial_edges(execution_trace, nodes) -> None:
    """Pre-populate message_to from already-built spatial edges.

    Called after the round topology is wired so the visualizer can draw arrows
    even for agents whose prompt/response haven't been recorded yet.
    RAG↔Researcher links are written inside researcher.py and are skipped here.
    """
    for node in nodes.values():
        agent_name = node.agent_name
        if agent_name not in execution_trace.trace[-1]:
            continue
        for succ in node.spatial_successors:
            if succ.agent_name in execution_trace.trace[-1]:
                execution_trace.trace[-1][agent_name]["message_to"].append(succ.agent_name)
