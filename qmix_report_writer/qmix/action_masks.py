"""
Per-phase / per-role action masks (upgrade-plan Stage 4.1, report 2.4).

The masks encode the handcrafted topology knowledge as *constraints* instead
of fixed choices: the QMIX policy chooses freely among the actions a careful
hand design would even consider. All rules live in this one module.

Rules (v1):
  * self-targeting selective_query is always masked (a self-edge is a no-op
    that would waste one of the agent's actions — absorbs report bug 0.7);
  * `append` is masked in every trained phase — PLANNING/RESEARCH because
    nothing may reach the report before drafting (report 2.4), DRAFTING
    because the write round stays scripted per open-decision OD-1 choice (a);
  * `append` is always masked for the Reviewer (its critiques must never
    become report prose);
  * `no_op` is masked for agents the current round hard-requires
    (RoundTopology.required_agents), so a random policy cannot produce a dead
    round during exploration.

The mask is a per-agent boolean list over qmix.agent_network.ACTION_NAMES
(True = selectable). Only ACTING agents get masks — the Collector never
selects actions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from qmix_report_writer.handcrafted_graph.phases import PhaseType
from qmix_report_writer.qmix.agent_network import ACTION_NAMES, NUM_ACTIONS

# Acting agents in roster order — index j is the target of selective_query
# action (2 + j). Must match the roster order used by the controller.
DEFAULT_ACTING_AGENTS: List[str] = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer"]

_APPEND_IDX = ACTION_NAMES.index("append")
_NO_OP_IDX = ACTION_NAMES.index("no_op")
_SELECTIVE_BASE = 2  # actions 2..2+len(acting)-1 are selective_query targets

# Phases in which `append` is masked for everyone. PLANNING/RESEARCH: nothing
# may reach the report yet. DRAFTING: OD-1 (a) — the write round is scripted
# by the environment; flip DRAFTING out of this set to experiment with
# policy-triggered appends (OD-1 (b)).
_APPEND_MASKED_PHASES = frozenset({
    PhaseType.PLANNING, PhaseType.RESEARCH, PhaseType.DRAFTING,
})


def mask(
    phase,
    agent_name: str,
    round_ctx: Optional[Dict] = None,
    acting_agents: Optional[Sequence[str]] = None,
) -> List[bool]:
    """Valid-action mask for one acting agent in one round.

    Args:
        phase: PhaseType of the current round.
        agent_name: The acting agent the mask is for.
        round_ctx: Optional round context; recognised keys:
            "required_agents": set of agent names the round hard-requires
                               (their no_op is masked).
        acting_agents: Acting-agent roster in selective-target order; defaults
                       to DEFAULT_ACTING_AGENTS.

    Returns:
        List of NUM_ACTIONS booleans, True = action selectable.
    """
    acting = list(acting_agents) if acting_agents is not None else DEFAULT_ACTING_AGENTS
    valid = [True] * NUM_ACTIONS

    # Self-targeting selective_query is never valid.
    for j, name in enumerate(acting):
        if name == agent_name:
            valid[_SELECTIVE_BASE + j] = False
    # Selective slots beyond the roster size (roster smaller than 4) are invalid.
    for j in range(len(acting), 4):
        valid[_SELECTIVE_BASE + j] = False

    # Append rules.
    if phase in _APPEND_MASKED_PHASES or agent_name == "Reviewer":
        valid[_APPEND_IDX] = False

    # Hard-required agents may not skip.
    required = (round_ctx or {}).get("required_agents") or ()
    if agent_name in required:
        valid[_NO_OP_IDX] = False

    return valid


def masks_for_round(
    phase,
    acting_agents: Sequence[str],
    round_ctx: Optional[Dict] = None,
) -> List[List[bool]]:
    """Masks for every acting agent, in roster order (one row per agent)."""
    return [
        mask(phase, name, round_ctx=round_ctx, acting_agents=acting_agents)
        for name in acting_agents
    ]
