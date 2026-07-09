"""
QMIX pipeline runners (upgrade-plan Stage 4.6).

Both modes run the SAME graph as the handcrafted pipeline — HandcraftedGraph
with a QMIXRoundController plugged into the controller seam:

  * run_qmix (inference): greedy policy, full phase list, finalization on →
    the same artifacts as run_handcrafted (markdown, optional PDF, trace,
    citations, bibliography, abstract).
  * run_qmix_train (training): ε-greedy policy, PLANNING→RESEARCH→DRAFTING
    only, max_validation_attempts=0, finalize=False (decision D6). After each
    episode the recorded transitions go to the replay buffer and
    `train_steps_per_episode` gradient steps run (once `min_buffer_episodes`
    is met). Every `eval_interval` episodes a GREEDY eval episode measures the
    policy (never buffered) and drives the best checkpoint; every episode is
    logged to <output_root>/qmix_train_log.jsonl with the full reward
    decomposition (training_eval plan Stage 4).

The report evaluator (grounded LLM judges + reward composition, package
module qmix_report_writer/evaluation) is built here by default and remains
injectable for tests and custom scorers.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from qmix_report_writer.evaluation import ReportEvaluator
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
from qmix_report_writer.utils.globals import ExecutionTrace, ReportState
from qmix_report_writer.utils.report_filter import filter_meta_commentary

logger = logging.getLogger("qmix.runner")

# Training episodes never run the correction phases (decision D6).
TRAINING_PHASES = [PLANNING_PHASE, RESEARCH_PHASE, DRAFTING_PHASE]

_FALLBACK_AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]

QMIX_TRACE_FILENAME = "qmix_trace.json"
TRAIN_LOG_FILENAME = "qmix_train_log.jsonl"


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
        target_update_interval=int(q.get("target_update_interval", 100)),
        buffer_capacity=int(q.get("buffer_capacity", 5000)),
        batch_size=int(q.get("batch_size", 32)),
        grad_clip=float(q.get("grad_clip", 10.0)),
        device=device,
    )


def _save_trace(filename: str = QMIX_TRACE_FILENAME) -> None:
    ExecutionTrace.instance().save_trace(str(get_output_root() / filename))


# ---------------------------------------------------------------------------
# Training-loop engineering helpers (training_eval plan Stage 4)
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed torch/numpy/random (plan 4.4). Logged in the JSONL header."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _append_train_log(record: dict, path: str) -> None:
    """Append one JSONL record to the training log (plan 4.3)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _best_path(save_path: str) -> str:
    """Best-eval checkpoint lives next to the latest one (plan 4.2)."""
    root, ext = os.path.splitext(save_path)
    return f"{root}_best{ext or '.pt'}"


def _chunk_decomposition(controller) -> list:
    """Per-event ChunkScore decomposition for the JSONL log (plan 4.3)."""
    return [{
        "score": cs.score,
        "rubric_mean": cs.rubric_mean,
        "logical_soundness": cs.logical_soundness,
        "verifiability": cs.verifiability_score,
        "technical_precision": cs.technical_precision,
        "info_density": cs.info_density,
        "hallucination": cs.hallucination_flag,
        "grounding_ratio": cs.grounding_ratio,
        "claims_sup_unsup_contra": [cs.n_supported, cs.n_unsupported,
                                    cs.n_contradicted],
    } for cs in getattr(controller, "chunk_scores", [])]


def _macro_decomposition(macro) -> Optional[dict]:
    if macro is None:
        return None
    return {
        "score": macro.score,
        "subject_coverage": macro.subject_coverage,
        "global_flow": macro.global_flow,
        "structural_score": macro.structural_score,
        "tone_consistency": macro.tone_consistency,
        "redundancy_avoidance": macro.redundancy_avoidance,
    }


def _episode_stats(controller, aborted: Optional[str]) -> dict:
    """The reward decomposition shared by training and eval records."""
    episode = controller.episode
    return {
        "aborted": aborted,
        "steps": episode.length,
        "tokens": episode.total_tokens,
        "total_reward": episode.total_reward,
        "step_rewards": [round(s.team_reward, 6) for s in episode.steps],
        "report_chars": len(ReportState.instance().content),
        "chunk_scores": _chunk_decomposition(controller),
        "macro": _macro_decomposition(controller.macro_score),
        "judge_failures": controller.judge_failures,
    }


async def _run_eval_episode(
    trainer: QMIXTrainer,
    agent_names: List[str],
    llm_name: str,
    task: str,
    evaluator,
    reward_cfg: dict,
    execution_trace: bool = False,
    max_tries: int = 3,
    max_time: int = 300,
) -> dict:
    """One GREEDY episode (ε=0, D6 shape), scored but NOT pushed to the
    buffer (plan 4.2) — the policy's actual quality, exploration-free.
    """
    _reset_run_state()
    controller = QMIXRoundController(
        trainer,
        agent_names,
        train=True,   # record + score; the episode is simply never pushed
        epsilon=0.0,
        evaluator=evaluator,
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
        await graph.arun(
            {"task": task},
            max_tries=max_tries,
            max_time=max_time,
            max_validation_attempts=0,
            finalize=False,
        )
    except NoCorpusCoverageError as exc:
        aborted = str(exc)
        logger.warning(f"Eval episode aborted: {exc}")
    finally:
        await controller.on_run_end()
        if execution_trace:
            _save_trace("qmix_trace_eval.json")

    return {"task": task, **_episode_stats(controller, aborted)}


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
    evaluator=None,
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
        evaluator: ReportEvaluator-like scorer; None builds the default
               grounded evaluator from config (injectable for tests).
    """
    cfg = get_config()
    qcfg = cfg.get("qmix", {}) or {}
    tcfg = qcfg.get("training", {}) or {}
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
    train_steps_per_episode = int(tcfg.get("train_steps_per_episode", 4))
    min_buffer_episodes = int(tcfg.get("min_buffer_episodes", 8))
    eval_interval = int(tcfg.get("eval_interval", 25))
    persist_buffer = bool(tcfg.get("persist_buffer", True))
    seed = tcfg.get("seed")
    if seed is not None:
        _seed_everything(int(seed))
        logger.info(f"Seeded torch/numpy/random with {seed}")

    trainer = build_trainer(len(agent_names), device=device)

    # best-checkpoint metric (plan 4.2): the greedy eval score; before the
    # first eval, the moving average of the last 5 training rewards.
    epsilon = eps_start
    start_episode = 0
    best_metric = float("-inf")
    had_eval = False

    if resume_path:
        meta = trainer.load(resume_path)
        if meta.get("epsilon") is not None:
            epsilon = float(meta["epsilon"])
        if meta.get("episode_idx"):
            start_episode = int(meta["episode_idx"])
        if meta.get("best_eval") is not None:
            best_metric = float(meta["best_eval"])
            had_eval = True  # never mix a restored metric with the fallback
        buffer_path = resume_path + ".buffer.pt"
        if persist_buffer and os.path.exists(buffer_path):
            trainer.replay_buffer.load(buffer_path)
            logger.info(
                f"Restored replay buffer: {len(trainer.replay_buffer)} episodes"
            )
        logger.info(
            f"Resuming at episode {start_episode} (eps={epsilon:.3f}, "
            f"best={best_metric if had_eval else 'n/a'})"
        )
    if evaluator is None:
        evaluator = ReportEvaluator()

    config_echo = {"qmix": qcfg, "reward": reward_cfg}
    log_path = str(get_output_root() / TRAIN_LOG_FILENAME)
    _append_train_log({
        "type": "header",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "num_episodes": num_episodes,
        "start_episode": start_episode,
        "llm": llm_name,
        "agents": agent_names,
        "resume": resume_path,
        "save_path": save_path,
        "config": config_echo,
    }, log_path)

    def _save_checkpoint(path: str, episode_idx: int, with_buffer: bool) -> None:
        trainer.save(path, epsilon=epsilon, episode_idx=episode_idx,
                     best_eval=(best_metric if best_metric != float("-inf")
                                else None),
                     config_echo=config_echo)
        if with_buffer and persist_buffer:
            trainer.replay_buffer.save(path + ".buffer.pt")

    eps_decay = (eps_start - eps_end) / max(num_episodes, 1)
    recent_rewards: List[float] = []
    train_start = time.time()

    for ep_idx in range(start_episode, num_episodes):
        ep_start = time.time()
        task = tasks[ep_idx % len(tasks)]

        _reset_run_state()
        controller = QMIXRoundController(
            trainer,
            agent_names,
            train=True,
            epsilon=epsilon,
            evaluator=evaluator,
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
                # Per-episode file (plan 5.3.1): an overwritten single trace
                # made the first live abort undiagnosable once episode 2 ran.
                _save_trace(f"qmix_trace_ep{ep_idx + 1}.json")

        episode = controller.episode
        stats = _episode_stats(controller, aborted)
        if episode.steps:
            trainer.replay_buffer.push(episode)

        # Replay ratio + warmup (plan 4.1): k gradient steps per episode
        # once the buffer holds enough episodes.
        losses: List[float] = []
        if len(trainer.replay_buffer) >= min_buffer_episodes:
            for _ in range(train_steps_per_episode):
                info = trainer.train_step()
                if info:
                    losses.append(round(info["loss"], 6))

        epsilon_used = epsilon
        epsilon = max(eps_end, epsilon - eps_decay)

        # Greedy eval episode (plan 4.2): fixed anchor task (tasks[0]) so the
        # eval score is comparable across the whole run; never buffered.
        eval_result: Optional[dict] = None
        if eval_interval and (ep_idx + 1) % eval_interval == 0:
            logger.info(f"Eval episode after episode {ep_idx + 1} (greedy)")
            eval_result = await _run_eval_episode(
                trainer, agent_names, llm_name, tasks[0], evaluator,
                reward_cfg, execution_trace=execution_trace,
                max_tries=max_tries, max_time=max_time,
            )

        # Best checkpoint (plan 4.2): eval score once evals exist; before the
        # first one, the moving average of the last 5 training rewards.
        total_reward = episode.total_reward
        recent_rewards.append(total_reward)
        candidate = metric_src = None
        if eval_result is not None:
            had_eval = True
            candidate, metric_src = eval_result["total_reward"], "eval"
        elif not had_eval:
            window = recent_rewards[-5:]
            candidate, metric_src = sum(window) / len(window), "train_ma5"
        if save_path and candidate is not None and candidate > best_metric:
            best_metric = candidate
            _save_checkpoint(_best_path(save_path), ep_idx + 1, with_buffer=False)
            logger.info(
                f"New best checkpoint ({metric_src}={candidate:.4f}) → "
                f"{_best_path(save_path)}"
            )
        if save_path and save_interval and (ep_idx + 1) % save_interval == 0:
            _save_checkpoint(save_path, ep_idx + 1, with_buffer=True)

        # ── Episode stats ────────────────────────────────────────────────
        ep_elapsed = time.time() - ep_start
        total_elapsed = time.time() - train_start
        done_count = ep_idx - start_episode + 1
        remaining = (total_elapsed / done_count) * (num_episodes - ep_idx - 1)

        _append_train_log({
            "type": "episode",
            "episode": ep_idx + 1,
            "task": task,
            "epsilon": round(epsilon_used, 4),
            **stats,
            "losses": losses,
            "training_step": trainer.training_step,
            "buffer_episodes": len(trainer.replay_buffer),
            "wall_time_s": round(ep_elapsed, 1),
            "eval": eval_result,
            "best_metric": (best_metric if best_metric != float("-inf")
                            else None),
        }, log_path)

        chunk = controller.last_chunk_score
        macro = controller.macro_score
        chunk_str = f"{chunk.score:.2f}" if chunk else "n/a"
        macro_str = f"{macro.score:.2f}" if macro else "n/a"
        loss_str = f"loss={losses[-1]:.4f}" if losses else "loss=n/a"
        status = f"ABORTED ({aborted[:60]}…)" if aborted else \
            ReportState.instance().progress[:100].replace("\n", " ")

        print(f"\n--- Episode {ep_idx + 1}/{num_episodes} [{ep_elapsed:.1f}s] ---")
        print(f"  Task:   {task[:80]}...")
        print(f"  Output: {status}...")
        print(f"  chunk={chunk_str} | macro={macro_str} | "
              f"judge_fails={controller.judge_failures} | "
              f"total reward={total_reward:.3f} | "
              f"steps={episode.length} | tokens={episode.total_tokens} | "
              f"eps={epsilon:.3f} | {loss_str} ({len(losses)} steps)")
        if eval_result is not None:
            print(f"  EVAL: reward={eval_result['total_reward']:.3f} | "
                  f"steps={eval_result['steps']} | "
                  f"judge_fails={eval_result['judge_failures']} | "
                  f"best={best_metric:.3f}")
        print(f"  Elapsed: {total_elapsed:.0f}s | ETA: {remaining:.0f}s "
              f"({remaining / 60:.1f}min)")

    if save_path:
        _save_checkpoint(save_path, num_episodes, with_buffer=True)

    print()
    print("=" * 60)
    print("  QMIX TRAINING COMPLETE")
    print("=" * 60)
    best_str = f"{best_metric:.4f}" if best_metric != float("-inf") else "n/a"
    print(f"  Best metric:  {best_str} ({'eval' if had_eval else 'train_ma5'})")
    print(f"  Avg Reward:   {trainer.replay_buffer.avg_reward:.4f}")
    print(f"  Avg Tokens:   {trainer.replay_buffer.avg_tokens:.0f}")
    print(f"  Steps:        {trainer.training_step}")
    if save_path:
        print(f"  Model saved:  {save_path} (best: {_best_path(save_path)})")
        print(f"  Train log:    {log_path}")
    print("=" * 60)

    return trainer
