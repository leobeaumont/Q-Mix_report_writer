"""
Round scheduler — decides which agents participate in a given round.

The scheduler separates two concerns:
  1. Participation: which agents execute this round (required + optional).
  2. Skip logic: optional agents may be excluded to avoid wasted inference.

Skip decision strategies (in order of sophistication):
  - ALWAYS_INCLUDE  : include all optional agents unconditionally (default,
                      safest for correctness; worst for token efficiency).
  - TEMPORAL_HEURISTIC : include an optional agent only if it produced output
                      in a previous round (has something to build on).
  - LLM_GATECHECK  : run a single lightweight LLM call asking the agent
                      "should you participate?" before its full execution
                      (most adaptive; adds one cheap call per optional agent).

The active strategy is set at construction time. LLM_GATECHECK requires an
`llm` instance to be passed in; otherwise it falls back to TEMPORAL_HEURISTIC.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Dict, Optional, Set

from qmix_report_writer.graph.node import Node

logger = logging.getLogger("handcrafted_graph.scheduler")


class SkipStrategy(Enum):
    ALWAYS_INCLUDE = "always_include"
    TEMPORAL_HEURISTIC = "temporal_heuristic"
    LLM_GATECHECK = "llm_gatecheck"


# System prompt used for the lightweight skip gate-check.
_GATECHECK_SYSTEM = (
    "You are a participation decision engine for a multi-agent pipeline. "
    "Based on the current context, decide whether your agent role adds value "
    "this round. Answer with exactly one word: EXECUTE or SKIP."
)


class RoundScheduler:
    """Determines which agents participate in each round of a phase.

    Args:
        nodes: The full node dict (node_id → Node) from HandcraftedGraph.
        collector_id: Node id of the Collector agent.
        skip_strategy: How optional agents decide to skip.
        llm: LLM instance for LLM_GATECHECK strategy (optional).
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        collector_id: Optional[str],
        skip_strategy: SkipStrategy = SkipStrategy.ALWAYS_INCLUDE,
        llm=None,
    ) -> None:
        self.nodes = nodes
        self.collector_id = collector_id
        self.skip_strategy = skip_strategy
        self.llm = llm

        # Fall back gracefully when LLM_GATECHECK is requested but no llm given.
        if self.skip_strategy == SkipStrategy.LLM_GATECHECK and self.llm is None:
            logger.warning(
                "LLM_GATECHECK requested but no llm provided; "
                "falling back to TEMPORAL_HEURISTIC."
            )
            self.skip_strategy = SkipStrategy.TEMPORAL_HEURISTIC

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_active_agents(
        self,
        topology,  # RoundTopology
        round_idx: int,
        task_input: Optional[Dict] = None,
    ) -> Set[str]:
        """Return the set of agent names that will execute this round.

        Required agents are always included. Optional agents are evaluated
        according to the configured skip strategy.
        """
        active: Set[str] = set(topology.required_agents)

        for agent_name in topology.optional_agents:
            node = self._get_node_by_name(agent_name)
            if node is None:
                logger.debug(f"Optional agent '{agent_name}' not found; skipping.")
                continue

            if await self._should_participate(node, agent_name, round_idx, task_input):
                active.add(agent_name)
            else:
                logger.debug(
                    f"Round {round_idx}: '{agent_name}' decided to SKIP "
                    f"(strategy={self.skip_strategy.value})."
                )

        return active

    # ------------------------------------------------------------------
    # Skip evaluation
    # ------------------------------------------------------------------

    async def _should_participate(
        self,
        node: Node,
        agent_name: str,
        round_idx: int,
        task_input: Optional[Dict],
    ) -> bool:
        if self.skip_strategy == SkipStrategy.ALWAYS_INCLUDE:
            return True

        if self.skip_strategy == SkipStrategy.TEMPORAL_HEURISTIC:
            return self._temporal_heuristic(node)

        if self.skip_strategy == SkipStrategy.LLM_GATECHECK:
            return await self._llm_gatecheck(node, agent_name, task_input)

        return True

    def _temporal_heuristic(self, node: Node) -> bool:
        """Include the optional agent if it has produced meaningful output before.

        Agents that explicitly held last round (output starts with a hold
        signal) are treated the same as agents that never ran — they are
        excluded from the next round unless the seed condition fires.

        Hold signals recognised:
          [HOLD]               — agent chose to hold (e.g. Researcher in REVISION
                                 when DataAnalyst flagged no gaps).
          [RESEARCH_EXHAUSTED] — RAG returned no documents for the query.

        On round 0 of every phase all memory is cleared, so no agent has
        prior output yet — the seed condition includes all optional agents.
        """
        outputs = node.last_memory.get("outputs") or []
        last_output = str(outputs[-1]).strip() if outputs else ""
        has_prior_output = (
            bool(last_output)
            and not last_output.startswith("[HOLD]")
            and not last_output.startswith("[RESEARCH_EXHAUSTED]")
        )
        # Seed condition: include every optional agent on the first round of
        # a phase when no agent has run yet (all memory was just cleared).
        any_agent_has_output = any(
            bool(n.last_memory.get("outputs")) for n in self.nodes.values()
        )
        if not any_agent_has_output:
            return True
        return has_prior_output

    async def _llm_gatecheck(
        self,
        node: Node,
        agent_name: str,
        task_input: Optional[Dict],
    ) -> bool:
        """Ask the LLM whether this agent should participate this round.

        Uses a minimal prompt to keep the cost at ~50 tokens per optional agent.
        Returns True on any error (fail-open).
        """
        from qmix_report_writer.utils.globals import ReportState

        report_progress = ReportState.instance().progress
        team_objective = ReportState.instance().task
        prior_output = ""
        if node.last_memory.get("outputs"):
            prior_output = str(node.last_memory["outputs"][-1])[:300]

        user_prompt = (
            f"Agent role: {node.role}\n"
            f"Team objective: {team_objective}\n"
            f"Report progress: {report_progress[:200]}\n"
        )
        if prior_output:
            user_prompt += f"Your last output: {prior_output}\n"
        user_prompt += "Should you participate this round? Answer EXECUTE or SKIP."

        messages = [
            {"role": "system", "content": _GATECHECK_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response: str = await self.llm.agen(messages)
            return "SKIP" not in response.upper()
        except Exception as exc:
            logger.warning(f"Gatecheck LLM call failed for '{agent_name}': {exc}. Defaulting to EXECUTE.")
            return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_node_by_name(self, agent_name: str) -> Optional[Node]:
        for node in self.nodes.values():
            if node.agent_name == agent_name:
                return node
        return None
