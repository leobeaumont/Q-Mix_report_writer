"""
QMIX pipeline runners (upgrade-plan Stage 4.6).

Both modes run the SAME graph as the handcrafted pipeline — HandcraftedGraph
with a QMIXRoundController plugged into the controller seam:

  * run_qmix (inference): greedy policy, full phase list, finalization on →
    the same artifacts as run_handcrafted (markdown, optional PDF, trace,
    citations, bibliography, abstract).
  * run_qmix_train (training): ε-greedy policy, PLANNING→RESEARCH→DRAFTING
    only, max_validation_attempts=0, finalize=False (decision D6). After each
    episode the recorded transitions go to the replay buffer and train_step()
    runs.

The report-quality scorer (LLM judges) is INJECTED as score_fn — the package
does not import the repo-level experiments/ module; the CLI wires it in. The
evaluator itself is scheduled for a separate rework.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Sequence, Tuple

from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph, NoCorpusCoverageError
from qmix_report_writer.handcrafted_graph.phases import (
    DRAFTING_PHASE, PLANNING_PHASE, RESEARCH_PHASE,
)
from qmix_report_writer.handcrafted_graph.runner import _reset_singletons
from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
from qmix_report_writer.qmix.observations import get_obs_dim, get_state_dim
from qmix_report_writer.qmix.qmix_controller import QMIXRoundController
from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
from qmix_report_writer.utils.config import get_config, get_output_root
from qmix_report_writer.utils.globals import ExecutionTrace, ReportState, Score
from qmix_report_writer.utils.report_filter import filter_meta_commentary

logger = logging.getLogger("qmix.runner")

# Training episodes never run the correction phases (decision D6).
TRAINING_PHASES = [PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE]

_FALLBACK_AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]

QMIX_TRACE_FILENAME = "qmix_trace.json"


def _agent_roster() -> List[str]:
    """Roster from config (the CORRECT `agent_configs` key — legacy bug 0.1)."""
    cfg = get_config()
    return (
        cfg.get("agent_configs", {}).get("redacting", {}).get("agents")
        or list(_FALLBACK_AGENTS)
    )


def _reset_run_state() -> None:
    """Per-episode/run singleton reset (includes SourceBuffer — legacy bug 0.6)."""
    _reset_singletons()
    ExecutionTrace.instance().reset()


def build_trainer(n_agents: int, device: str = "cpu") -> QMIXTrainer:
    """QMIXTrainer wired from config; dims come from the observation builder."""
    q = get_config().get("qmix", {}) or {}
    reward = get_config().get("reward", {}) or {}
    return QMIXTrainer(
        n_agents=n_agents,
        obs_dim=get_obs_dim(),
        state_dim=get_state_dim(n_agents),
        n_actions=NUM_ACTIONS,
        gnn_hidden_dim=int(q.get("gnn_hidden_dim", 128)),
        gnn_layers=int(q.get("gnn_layers", 2)),
        rnn_hidden_dim=int(q.get("rnn_hidden_dim", 128)),
        mixing_hidden_dim=int(q.get("mixing_hidden_dim", 64)),
        lr=float(q.get("lr", 5e-4)),
        gamma=float(q.get("gamma", 0.99)),
        target_update_interval=int(q.get("target_update_interval", 200)),
        buffer_capacity=int(q.get("buffer_capacity", 5000)),
        batch_size=int(q.get("batch_size", 32)),
        grad_clip=float(q.get("grad_clip", 10.0)),
        length_weight=float(reward.get("length_weight", 0.1)),
        report_quality_weight=float(reward.get("report_quality_weight", 1.0)),
        device=device,
    )


def _save_trace() -> None:
    ExecutionTrace.instance().save_trace(str(get_output_root() / QMIX_TRACE_FILENAME))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

async def run_qmix(
    task: str,
    model_path: Optional[str] = None,
    llm_name: Optional[str] = None,
    agent_names: Optional[List[str]] = None,
    execution_trace: bool = False,
    max_tries: int = 3,
    max_time: int = 300,
    save_output: bool = True,
    export_pdf: bool = True,
    device: str = "cpu",
) -> Tuple[List[str], int]:
    """Generate one report with the (trained) QMIX policy — full pipeline.

    Mirrors run_handcrafted: greedy action selection, all phases including
    SECTION_REVIEW/VALIDATION, finalization on, same artifact path.
    """
    cfg = get_config()
    if llm_name is None:
        llm_name = cfg.get("llm", {}).get("default_model", "qwen3:8b")
    if agent_names is None:
        agent_names = _agent_roster()

    _reset_run_state()

    trainer = build_trainer(len(agent_names), device=device)
    if model_path:
        trainer.load(model_path)
    else:
        logger.warning("No model_path given — running with an UNTRAINED policy.")

    controller = QMIXRoundController(trainer, agent_names, train=False)
    graph = HandcraftedGraph(
        llm_name=llm_name,
        agent_names=agent_names,
        execution_trace=execution_trace,
        controller=controller,
    )

    logger.info(f"Starting QMIX inference run | task='{task[:80]}...' | llm={llm_name}")
    try:
        answers, total_tokens = await graph.arun(
            {"task": task}, max_tries=max_tries, max_time=max_time
        )
    finally:
        if execution_trace:
            _save_trace()

    answers = [filter_meta_commentary(a) for a in answers]

    if save_output and answers:
        from qmix_report_writer.utils.report_export import save_raw_report
        run_dir = save_raw_report(task=task, report=answers[0])
        logger.info(f"Raw report saved to {run_dir}")
        pdf_path = None
        if export_pdf:
            try:
                from qmix_report_writer.utils.markdown_to_latex import convert_run_dir
                from qmix_report_writer.utils.compile_pdf import compile_run_dir
                convert_run_dir(run_dir)
                pdf_path = compile_run_dir(run_dir)
                logger.info(f"Report PDF compiled to {pdf_path}")
            except Exception as exc:
                logger.warning(
                    f"PDF export failed ({exc}). Raw markdown is available at {run_dir}."
                )
        print(f"\nReport saved to: {run_dir}")
        if pdf_path is not None:
            print(f"PDF: {pdf_path}")

    return answers, total_tokens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

async def run_qmix_train(
    tasks: Sequence[str],
    score_fn: Callable,
    llm_name: Optional[str] = None,
    agent_names: Optional[List[str]] = None,
    num_episodes: Optional[int] = None,
    device: str = "cpu",
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    execution_trace: bool = False,
    max_tries: int = 3,
    max_time: int = 300,
) -> QMIXTrainer:
    """Train the QMIX policy on report-writing episodes (decision D6 shape).

    Args:
        tasks: Report subjects, cycled across episodes (injected by the CLI —
               the package does not import the repo-level datasets module).
        score_fn: Async () -> float report scorer (the LLM judges; injected by
               the CLI from experiments/eval.py until the evaluator rework).
    """
    cfg = get_config()
    tcfg = (cfg.get("qmix", {}) or {}).get("training", {}) or {}
    reward_cfg = cfg.get("reward", {}) or {}
    if llm_name is None:
        llm_name = cfg.get("llm", {}).get("default_model", "qwen3:8b")
    if agent_names is None:
        agent_names = _agent_roster()
    if num_episodes is None:
        num_episodes = int(tcfg.get("num_episodes", 500))
    eps_start = float(tcfg.get("epsilon_start", 1.0))
    eps_end = float(tcfg.get("epsilon_end", 0.05))
    save_interval = int(tcfg.get("save_interval", 50))

    trainer = build_trainer(len(agent_names), device=device)
    if resume_path:
        trainer.load(resume_path)

    epsilon = eps_start
    eps_decay = (eps_start - eps_end) / max(num_episodes, 1)
    best_reward = float("-inf")
    train_start = time.time()

    for ep_idx in range(num_episodes):
        ep_start = time.time()
        task = tasks[ep_idx % len(tasks)]

        _reset_run_state()
        controller = QMIXRoundController(
            trainer,
            agent_names,
            train=True,
            epsilon=epsilon,
            score_fn=score_fn,
            length_goal=int(reward_cfg.get("length_goal", 25000)),
            length_sigma=int(reward_cfg.get("length_sigma", 8500)),
        )
        graph = HandcraftedGraph(
            llm_name=llm_name,
            agent_names=agent_names,
            execution_trace=execution_trace,
            phases=TRAINING_PHASES,
            controller=controller,
        )

        aborted: Optional[str] = None
        try:
            answers, _ = await graph.arun(
                {"task": task},
                max_tries=max_tries,
                max_time=max_time,
                max_validation_attempts=0,
                finalize=False,
            )
        except NoCorpusCoverageError as exc:
            # Episode ends; buffered steps flush at reward 0 (plan 4.4).
            aborted = str(exc)
            logger.warning(f"Episode {ep_idx + 1} aborted: {exc}")
        finally:
            await controller.on_run_end()  # idempotent — arun also calls it
            if execution_trace:
                _save_trace()

        episode = controller.episode
        if episode.steps:
            trainer.replay_buffer.push(episode)
        train_info = trainer.train_step()

        epsilon = max(eps_end, epsilon - eps_decay)

        # ── Episode stats ────────────────────────────────────────────────
        ep_elapsed = time.time() - ep_start
        total_elapsed = time.time() - train_start
        remaining = (total_elapsed / (ep_idx + 1)) * (num_episodes - ep_idx - 1)
        score = Score.instance().current_score or 0.0
        total_reward = episode.total_reward
        loss_str = f"loss={train_info['loss']:.4f}" if train_info else "loss=n/a"
        status = f"ABORTED ({aborted[:60]}…)" if aborted else \
            ReportState.instance().progress[:100].replace("\n", " ")

        print(f"\n--- Episode {ep_idx + 1}/{num_episodes} [{ep_elapsed:.1f}s] ---")
        print(f"  Task:   {task[:80]}...")
        print(f"  Output: {status}...")
        print(f"  score={score:.2f} | total reward={total_reward:.3f} | "
              f"steps={episode.length} | tokens={episode.total_tokens} | "
              f"eps={epsilon:.3f} | {loss_str}")
        print(f"  Elapsed: {total_elapsed:.0f}s | ETA: {remaining:.0f}s "
              f"({remaining / 60:.1f}min)")

        if save_path and total_reward > best_reward:
            best_reward = total_reward
            trainer.save(save_path)
        if save_path and save_interval and (ep_idx + 1) % save_interval == 0:
            trainer.save(save_path)

    if save_path:
        trainer.save(save_path)

    print()
    print("=" * 60)
    print("  QMIX TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Reward:  {best_reward:.4f}")
    print(f"  Avg Reward:   {trainer.replay_buffer.avg_reward:.4f}")
    print(f"  Avg Tokens:   {trainer.replay_buffer.avg_tokens:.0f}")
    print(f"  Steps:        {trainer.training_step}")
    if save_path:
        print(f"  Model saved:  {save_path}")
    print("=" * 60)

    return trainer
