"""
RoundController — the seam between phase execution and round decision-making.

HandcraftedGraph consults a RoundController wherever it consulted the
RoundTopology tables + RoundScheduler before (upgrade-plan Stage 3, decision
D2). Exactly these spots are controller-driven:

  - the generic phase loop (PLANNING, RESEARCH, and the DRAFTING fallback
    loop used when no section outline was extracted),
  - the DRAFTING prep round (Round A),
  - the DRAFTING round-B *retry* branch (Round A produced no usable blueprint).

Scripted rounds NEVER consult the controller — they are environment rules,
not policy choices: the DRAFTING blueprint-reuse write round, all
SECTION_REVIEW rounds (review / revision / directive-bypass), and the
VALIDATION window + synthesis rounds.

Hooks: ``on_phase_start`` fires when a phase's execution begins;
``on_round_end`` fires after every PLANNING / RESEARCH / DRAFTING round
(controller-driven AND scripted write rounds) so a learning controller can
record transitions and detect report appends. Correction phases
(SECTION_REVIEW / VALIDATION) fire no hooks — QMIX training episodes never run
them (decision D6). ``on_run_end`` fires once at the end of ``arun``.

The default HandcraftedRoundController reproduces the pre-seam behavior
exactly: plans come from the phase's RoundTopology and the RoundScheduler's
skip strategy, with no per-agent actions. The QMIX controller (upgrade-plan
Stage 4.2) replaces plans with network-selected actions and uses the hooks for
episode recording and reward events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from qmix_report_writer.handcrafted_graph.scheduler import RoundScheduler, SkipStrategy


@dataclass
class RoundPlan:
    """What happens in one controller-driven round.

    Attributes:
        active_agents: Agent names that execute this round. Inactive agents
                       are skipped entirely (no LLM call, no edges).
        edges: Directed (sender_name, receiver_name) message edges. Only edges
               whose both endpoints are active get wired.
        actions: Optional per-agent action ids (agent_name -> action). When
                 set, the action is rendered into the agent's prompt context
                 block and recorded in the execution trace. None = handcrafted
                 mode (no action concept).
    """
    active_agents: Set[str]
    edges: List[Tuple[str, str]]
    actions: Optional[Dict[str, int]] = None


class RoundController:
    """Base controller. Subclasses decide participation and topology per round."""

    async def round_plan(
        self,
        phase,            # PhaseType
        round_idx: int,
        topology,         # RoundTopology — the phase table entry for this round
        nodes,            # Dict[node_id, Node]
        task_input,       # {"task": ...}
    ) -> RoundPlan:
        raise NotImplementedError

    def on_phase_start(self, phase) -> None:
        """Called when a phase's execution begins (before its first round)."""
        return None

    async def on_round_end(self, phase, round_idx: int) -> None:
        """Called after each PLANNING/RESEARCH/DRAFTING round completes."""
        return None

    async def on_run_end(self) -> None:
        """Called once at the end of arun(), after the last phase."""
        return None


class HandcraftedRoundController(RoundController):
    """Default controller: the phase tables + RoundScheduler, unchanged.

    One scheduler instance serves the whole run — RoundScheduler carries no
    phase state (it reads node memories at call time), so this is equivalent
    to the pre-seam behavior of building one scheduler per phase.
    """

    def __init__(
        self,
        nodes,
        collector_id: Optional[str],
        skip_strategy: SkipStrategy = SkipStrategy.ALWAYS_INCLUDE,
        llm=None,
    ) -> None:
        self.scheduler = RoundScheduler(
            nodes=nodes,
            collector_id=collector_id,
            skip_strategy=skip_strategy,
            llm=llm,
        )

    async def round_plan(self, phase, round_idx, topology, nodes, task_input) -> RoundPlan:
        active = await self.scheduler.get_active_agents(
            topology, round_idx, task_input=task_input
        )
        return RoundPlan(active_agents=active, edges=list(topology.edges), actions=None)
