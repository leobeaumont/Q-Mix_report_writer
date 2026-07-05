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

Reward events (reward v2 — training_eval plan 2.5, decisions TD1/TD2/OD-A):
fired from on_round_end when ReportState.additions actually GREW — never from
the chosen action. The appended section is scored by the injected evaluator
(grounded chunk audit + claim check); the composed reward lands on the EVENT
STEP only (OD-A) while earlier buffered steps flush at 0 — TD bootstrapping
propagates the credit. At run end the terminal macro score is added to the
final step. A judge failure skips the event (steps stay buffered; a parse or
transport hiccup never becomes a score). The flag-gated token penalty (TD2,
default off) is subtracted from every recorded step at finalization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from qmix_report_writer.evaluation.reward import (
    compose_event_reward, compose_terminal_reward, length_gaussian, token_penalty,
)
from qmix_report_writer.handcrafted_graph.controller import RoundController, RoundPlan
from qmix_report_writer.qmix.action_masks import masks_for_round
from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
from qmix_report_writer.qmix.observations import (
    build_adj, build_global_state, build_observations,
)
from qmix_report_writer.qmix.replay_buffer import Episode, EpisodeStep
from qmix_report_writer.utils.config import get_config
from qmix_report_writer.utils.globals import (
    CompletionTokens, PromptTokens, ReportState,
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
        trainer: QMIXTrainer whose agent network selects actions.
        agent_names: Full roster INCLUDING the Collector, in graph order.
        train: ε-greedy exploration + episode recording + reward events.
               False = greedy inference, nothing recorded.
        epsilon: Exploration rate used when train=True.
        evaluator: ReportEvaluator-like object (async score_chunk /
                  score_report, both may return None on judge failure).
                  None disables reward events (steps flush at reward 0 —
                  useful for offline dry runs).
        length_goal / length_sigma: Gaussian length-shaping parameters.
    """

    def __init__(
        self,
        trainer,
        agent_names: Sequence[str],
        train: bool = False,
        epsilon: float = 0.0,
        evaluator=None,
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
        self.evaluator = evaluator
        self.length_goal = length_goal
        self.length_sigma = length_sigma

        self.hidden: Optional[torch.Tensor] = None
        self.episode = Episode()
        self.judge_failures = 0
        self.last_chunk_score = None   # latest ChunkScore (runner stats/log)
        self.macro_score = None        # terminal MacroScore (runner stats/log)
        self._reward_cfg = dict(get_config().get("reward", {}) or {})
        # Length-shaping baseline, taken at construction (plan 2.5): the first
        # event's delta is measured from the episode's starting length.
        self._prev_length_gauss = length_gaussian(
            len(ReportState.instance().content), length_goal, length_sigma,
        )
        self._task = ""
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
        self._task = task or self._task  # episode task, used by the judges
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
        """Flush leftovers at 0, add the terminal macro reward, mark done.

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
            self._finalize_step(step, 0.0)
        self._step_buffer = []

        # Terminal macro reward (TD1): one whole-report judge call, added to
        # the final step — TD bootstrapping propagates it backward.
        report = ReportState.instance().content
        if self.train and self.evaluator is not None and report.strip() \
                and self.episode.steps:
            try:
                macro = await self.evaluator.score_report(
                    task=self._task,
                    outline=list(ReportState.instance().planned_sections),
                    report=report,
                )
            except Exception as exc:
                macro = None
                logger.warning(f"Evaluator raised during score_report: {exc}")
            if macro is None:
                self.judge_failures += 1
                logger.warning("Terminal macro judge failed — no terminal reward.")
            else:
                self.macro_score = macro
                self.episode.steps[-1].team_reward += compose_terminal_reward(
                    macro.score, self._reward_cfg,
                )
                logger.info(
                    f"Terminal macro: {macro.score:.3f} added to the final step."
                )

        if self.episode.steps:
            self.episode.steps[-1].done = True

    # ------------------------------------------------------------------
    # Reward event (reward v2 — plan 2.5)
    # ------------------------------------------------------------------

    def _finalize_step(self, step: EpisodeStep, base_reward: float) -> None:
        """Record a step with its reward minus the (flag-gated) token cost."""
        step.team_reward = base_reward - token_penalty(
            step.token_usage, self._reward_cfg,
        )
        self.episode.add_step(step)

    async def _reward_event(self) -> None:
        """Score the appended section; the reward lands on the EVENT STEP only.

        OD-A: the most recent recorded step (the one whose round led to the
        write) carries the composed reward; earlier buffered steps flush at 0
        and TD bootstrapping propagates the credit backward. A judge failure
        skips the event — buffered steps stay for the next one (plan 1.3).
        """
        if self.evaluator is None:
            # Recording without scoring (offline dry runs): steps stay in the
            # buffer and flush at reward 0 on run end.
            return
        if not self._step_buffer:
            logger.warning(
                "Reward event with an empty step buffer — skipped "
                "(no judge call made)."
            )
            return

        rs = ReportState.instance()
        section = rs.sections[-1] if rs.sections else None
        chunk = section["content"] if section else (
            rs.additions[-1] if rs.additions else rs.content
        )
        sources = list(section.get("sources") or []) if section else []
        try:
            chunk_score = await self.evaluator.score_chunk(
                chunk=chunk, sources=sources, task=self._task,
                context=rs.progress,
            )
        except Exception as exc:
            chunk_score = None
            logger.warning(f"Evaluator raised during score_chunk: {exc}")
        if chunk_score is None:
            self.judge_failures += 1
            logger.warning(
                f"Judge failure #{self.judge_failures} — reward event skipped, "
                f"{len(self._step_buffer)} buffered step(s) kept for the next "
                f"event."
            )
            return

        gauss = length_gaussian(len(rs.content), self.length_goal, self.length_sigma)
        reward = compose_event_reward(
            chunk_score.score, gauss - self._prev_length_gauss, self._reward_cfg,
        )
        self._prev_length_gauss = gauss
        self.last_chunk_score = chunk_score

        *earlier, event_step = self._step_buffer
        for step in earlier:
            self._finalize_step(step, 0.0)
        self._finalize_step(event_step, reward)
        self._step_buffer = []
        logger.info(
            f"Reward event: chunk={chunk_score.score:.3f} "
            f"(grounding={chunk_score.grounding_ratio}) reward={reward:.4f} "
            f"on the event step; {len(earlier)} earlier step(s) flushed at 0."
        )
