"""
QMIXRoundController — the QMIX policy plugged into the handcrafted graph
(upgrade-plan Stages 4.2 + 4.4).

Implements the RoundController seam (handcrafted_graph/controller.py): at each
controller-driven round it builds observations, selects one masked action per
acting agent through the trainer's networks, and translates the joint action
into a RoundPlan. The hooks record episode transitions and fire reward events.

Action → plan translation (see plan_from_actions):
    no_op            agent excluded from active_agents (no LLM call at all)
    broadcast_all    edges to every other ACTIVE acting agent
    selective_query  edge to the chosen acting agent (dropped if target no_op'd)
    aggregate_refine edges FROM every other active acting agent
    append           edge to the Collector (unreachable in v1: masked, OD-1a)

Environment invariant: a round whose RoundTopology hard-requires the Collector
(the DRAFTING round-B retry) keeps its table edges into the Collector, so the
write still happens regardless of what the policy chose.

Reward events (Stage 4.4, absorbs legacy bug 0.8): fired from on_round_end
when ReportState.additions actually GREW — never from the chosen action — so a
refused append (absence-marker gate, sentinel output) costs no judge calls and
never skews the score deltas. The reward is spread evenly over all steps
buffered since the previous event; leftovers flush at 0 on run end.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from qmix_report_writer.handcrafted_graph.controller import RoundController, RoundPlan
from qmix_report_writer.qmix.action_masks import masks_for_round
from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
from qmix_report_writer.qmix.observations import (
    build_adj, build_global_state, build_observations,
)
from qmix_report_writer.qmix.replay_buffer import Episode, EpisodeStep
from qmix_report_writer.utils.globals import (
    CompletionTokens, LengthGoal, PromptTokens, ReportState, Score,
)

logger = logging.getLogger("qmix_controller")


def plan_from_actions(
    actions: Dict[str, int],
    acting_agents: Sequence[str],
    topology,
) -> RoundPlan:
    """Translate a joint action (agent_name -> action id) into a RoundPlan."""
    active = {name for name, a in actions.items() if a != 0}
    edges: List[tuple] = []
    for name, a in actions.items():
        if a == 0:
            continue
        if a == 1:  # broadcast_all
            edges += [
                (name, other) for other in acting_agents
                if other != name and other in active
            ]
        elif 2 <= a <= 5:  # selective_query
            j = a - 2
            if j < len(acting_agents):
                target = acting_agents[j]
                if target != name and target in active:
                    edges.append((name, target))
        elif a == 6:  # aggregate_refine
            edges += [
                (other, name) for other in acting_agents
                if other != name and other in active
            ]
        elif a == 7:  # append
            edges.append((name, "Collector"))
            active.add("Collector")

    # Environment invariant: the DRAFTING retry round hard-requires the
    # Collector — keep the table's edges into it so the write always happens.
    if "Collector" in getattr(topology, "required_agents", ()):
        active.add("Collector")
        for sender, receiver in getattr(topology, "edges", ()):
            if receiver == "Collector" and sender in active:
                edges.append((sender, receiver))

    seen = set()
    unique_edges = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            unique_edges.append(edge)

    return RoundPlan(active_agents=active, edges=unique_edges, actions=dict(actions))


class QMIXRoundController(RoundController):
    """RoundController driven by the QMIX networks.

    Args:
        trainer: QMIXTrainer whose agent network selects actions (and whose
                 compute_reward combines the score deltas).
        agent_names: Full roster INCLUDING the Collector, in graph order.
        train: ε-greedy exploration + episode recording + reward events.
               False = greedy inference, nothing recorded.
        epsilon: Exploration rate used when train=True.
        score_fn: Async () -> float report-quality scorer (the LLM judges).
                  None disables reward events (steps flush at reward 0 —
                  useful for offline dry runs).
        length_goal / length_sigma: Gaussian length-score parameters
                  (env-side re-statement of the legacy length_score).
    """

    def __init__(
        self,
        trainer,
        agent_names: Sequence[str],
        train: bool = False,
        epsilon: float = 0.0,
        score_fn: Optional[Callable] = None,
        length_goal: int = 25000,
        length_sigma: int = 8500,
    ) -> None:
        if trainer.n_actions != NUM_ACTIONS:
            raise ValueError(
                f"Trainer built for {trainer.n_actions} actions, "
                f"action space has {NUM_ACTIONS}."
            )
        self.trainer = trainer
        self.agent_names = list(agent_names)
        self.acting_agents = [a for a in self.agent_names if a != "Collector"]
        self.train = train
        self.epsilon = epsilon
        self.score_fn = score_fn
        self.length_goal = length_goal
        self.length_sigma = length_sigma

        self.hidden: Optional[torch.Tensor] = None
        self.episode = Episode()
        self.judge_failures = 0
        self._step_buffer: List[EpisodeStep] = []
        self._pending: Optional[EpisodeStep] = None
        self._pending_tokens_before = 0.0
        self._last_additions = 0
        self._finalized = False

    # ------------------------------------------------------------------
    # RoundController interface
    # ------------------------------------------------------------------

    async def round_plan(self, phase, round_idx, topology, nodes, task_input) -> RoundPlan:
        task = str((task_input or {}).get("task", ""))
        obs = build_observations(nodes, task)
        adj = build_adj(nodes)

        if self.hidden is None:
            self.hidden = self.trainer.agent_network.init_hidden(len(nodes))

        round_ctx = {"required_agents": set(getattr(topology, "required_agents", ()))}
        mask_rows = masks_for_round(phase, self.acting_agents, round_ctx)
        mask_t = torch.tensor(mask_rows, dtype=torch.bool)

        actions_t, self.hidden = self.trainer.select_actions(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            self.hidden,
            epsilon=self.epsilon if self.train else 0.0,
            mask=mask_t,
        )

        # A mask violation here is a controller/trainer bug, never LLM noise —
        # fail loudly (upgrade-plan 4.8 requires this assertion).
        for i, action in enumerate(actions_t.tolist()):
            if not mask_rows[i][action]:
                raise RuntimeError(
                    f"Mask violated: agent {self.acting_agents[i]} chose "
                    f"action {action} in phase {phase} (mask {mask_rows[i]})."
                )

        actions = {
            name: int(a) for name, a in zip(self.acting_agents, actions_t.tolist())
        }
        logger.info(f"[{getattr(phase, 'value', phase)} r{round_idx}] actions={actions}")

        if self.train:
            self._pending = EpisodeStep(
                observations=obs.astype(np.float32),
                actions=actions_t.numpy(),
                rewards=np.zeros(len(self.acting_agents)),
                team_reward=0.0,  # set by the next reward event (or 0-flush)
                adj_matrix=adj.astype(np.float32),
                global_state=build_global_state(obs, adj).astype(np.float32),
                done=False,
                mask=np.asarray(mask_rows, dtype=bool),
            )
            self._pending_tokens_before = (
                PromptTokens.instance().value + CompletionTokens.instance().value
            )

        return plan_from_actions(actions, self.acting_agents, topology)

    async def on_round_end(self, phase, round_idx) -> None:
        if self._pending is not None:
            tokens_now = PromptTokens.instance().value + CompletionTokens.instance().value
            self._pending.token_usage = int(tokens_now - self._pending_tokens_before)
            self._step_buffer.append(self._pending)
            self._pending = None

        if not self.train:
            return

        # Reward trigger = the report actually grew (never the chosen action).
        additions = len(ReportState.instance().additions)
        if additions > self._last_additions:
            self._last_additions = additions
            await self._reward_event()

    async def on_run_end(self) -> None:
        """Flush buffered steps at reward 0 and mark the episode done.

        Idempotent: the runner also calls this from its finally-path so an
        aborted episode (e.g. NoCorpusCoverageError) still yields a
        well-formed episode.
        """
        if self._finalized:
            return
        self._finalized = True
        if self._pending is not None:
            self._step_buffer.append(self._pending)
            self._pending = None
        for step in self._step_buffer:
            step.team_reward = 0.0
            self.episode.add_step(step)
        self._step_buffer = []
        if self.episode.steps:
            self.episode.steps[-1].done = True

    # ------------------------------------------------------------------
    # Reward event (Stage 4.4)
    # ------------------------------------------------------------------

    def _length_score(self) -> float:
        """Gaussian length score over the current report length (env-side)."""
        length = len(ReportState.instance().content)
        return float(np.exp(-0.5 * ((length - self.length_goal) / self.length_sigma) ** 2))

    async def _reward_event(self) -> None:
        """Score the report, spread the delta reward over buffered steps."""
        if self.score_fn is None:
            # Recording without scoring (offline dry runs): steps stay in the
            # buffer and flush at reward 0 on run end.
            return
        # Judge failures skip the event instead of scoring 0 (plan 1.3): the
        # buffered steps stay for the next event; a parse/transport hiccup
        # must never enter the score history (defect A0.3).
        try:
            new_score = await self.score_fn()
        except Exception as exc:
            self.judge_failures += 1
            logger.warning(
                f"Judge failure #{self.judge_failures} — reward event skipped, "
                f"{len(self._step_buffer)} buffered step(s) kept for the next "
                f"event. ({exc})"
            )
            return
        Score.instance().update(new_score)
        LengthGoal.instance().update(self._length_score())
        reward = self.trainer.compute_reward(
            Score.instance().get_delta(), LengthGoal.instance().get_delta()
        )
        if not self._step_buffer:
            logger.warning("Reward event with an empty step buffer — dropped.")
            return
        share = reward / len(self._step_buffer)
        for step in self._step_buffer:
            step.team_reward = share
            self.episode.add_step(step)
        self._step_buffer = []
        logger.info(
            f"Reward event: reward={reward:.4f} spread over "
            f"{self.episode.length} recorded step(s) so far."
        )
