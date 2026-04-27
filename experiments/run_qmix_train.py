"""
QMIX Training Loop for Multi-Agent Communication Topology Optimization.

This script implements the centralized training phase:
1. Initialize QMIX trainer with agent networks and mixing network
2. For each episode:
   a. Sample a task from the dataset
   b. Agents select actions via epsilon-greedy using their Q-networks
   c. Execute the multi-agent graph with QMIX-selected topology
   d. Compute reward: delta_report_quality * report_quality_weight + delta_length_goal * length_weight 
   e. Store episode in replay buffer
   f. Train QMIX by minimizing TD loss
3. Save trained model for decentralized deployment
"""

import os
import sys
import argparse
import asyncio
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix.qmix_trainer import QMIXTrainer
from qmix.replay_buffer import Episode, EpisodeStep
from qmix.agent_network import NUM_ACTIONS
from graph.graph import QMIXGraph
from datasets import tasks
from utils.log import get_logger
from utils.globals import PromptTokens, CompletionTokens, ReportState, Score, LengthGoal, ExecutionTrace
from utils.config import get_config
from experiments.eval import length_score, report_score

logger = get_logger("qmix_train")


def get_obs_dim(n_agents, task_feature_dim=16, state_feature_dim=16):
    """Calculate observation dimension per agent."""
    agent_id_dim = 16
    extra_features = 3  # has_output, n_neighbors_ratio, token_ratio
    return task_feature_dim + state_feature_dim + agent_id_dim + extra_features  # 51


def get_state_dim(n_agents, obs_dim):
    """Calculate global state dimension for mixing network."""
    graph_stats = 3  # total_edges, density, n_agents
    return n_agents * obs_dim + graph_stats


async def run_episode(
    graph: QMIXGraph,
    trainer: QMIXTrainer,
    task_text: str,
    epsilon: float,
    max_rounds: int = 20,
    length_goal: int = 25000,
    length_sigma: int = 8500
) -> Episode:
    """Run a single episode of QMIX-driven multi-agent interaction."""
    episode = Episode()
    ReportState.instance().reset()
    Score.instance().reset()
    LengthGoal.instance().reset()
    ExecutionTrace.instance().reset()
    n_agents = graph.n_agents

    hidden = trainer.agent_network.init_hidden(n_agents)
    obs = torch.tensor(graph.get_observation_features(task_text), dtype=torch.float32)
    adj = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32)

    step = 0
    step_buffer = []
    round_pbar = tqdm(total=max_rounds, desc="Rounds", leave=False)
    while step < max_rounds and not graph.terminated:
        actions, hidden = trainer.select_actions(obs, adj, hidden, epsilon)

        tokens_before = PromptTokens.instance().value + CompletionTokens.instance().value
        answers, tokens_used = await graph.arun(
            {"task": task_text},
            num_rounds=1,
            actions=actions,
        )
        tokens_after = PromptTokens.instance().value + CompletionTokens.instance().value

        new_obs = torch.tensor(graph.get_observation_features(task_text), dtype=torch.float32)
        new_adj = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32)
        global_state = graph.get_global_state(task_text)

        step_data = EpisodeStep(
            observations=obs.numpy(),
            actions=actions.numpy(),
            rewards=np.zeros(n_agents),
            team_reward=0.0,  # placeholder, set after evaluation
            adj_matrix=adj.numpy(),
            global_state=global_state,
            done=(step == max_rounds - 1 or graph.terminated),
            token_usage=tokens_used,
        )
        step_buffer.append(step_data)

        if (actions == 14).any():  # If any agent appended this step
            # We spread the reward to all previous step until previous append action
            Score.instance().update(await report_score())
            LengthGoal.instance().update(length_score(length_goal, length_sigma))
            delta_report_score = Score.instance().get_delta()
            delta_length_goal = LengthGoal.instance().get_delta()
            reward = trainer.compute_reward(delta_report_score, delta_length_goal)
            share = reward / len(step_buffer)  # step_buffer is never empty here
            for previous_step in step_buffer:
                previous_step.team_reward = share
                episode.add_step(previous_step)
            step_buffer = []

        obs = new_obs
        adj = new_adj

        step += 1
        round_pbar.update()
    
    for previous_step in step_buffer:  # All unused steps in buffer are pushed with reward = 0
        episode.add_step(previous_step)

    ExecutionTrace.instance().save_trace()

    round_pbar.close()

    return episode, answers


async def train(args):
    """Main training loop."""

    agent_names = get_config().get("agent_config", {}
                             ).get("redacting", {}
                             ).get("agents", ["LeadArchitect", 
                                              "Researcher", 
                                              "DataAnalyst", 
                                              "TechnicalWriter", 
                                              "Reviewer", 
                                              "Collector"])
    n_agents = len(agent_names)

    print()
    print("=" * 60)
    print(f"  QMIX TRAINING: Redacting team")
    print("=" * 60)
    print(f"  LLM:        {args.llm_name}")
    print(f"  Agents:     {[a for a in agent_names]}")
    print(f"  Episodes:   {args.num_episodes}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Device:     {args.device}")
    print("=" * 60)
    print()

    obs_dim = get_obs_dim(n_agents)
    state_dim = get_state_dim(n_agents, obs_dim)

    trainer = QMIXTrainer(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=NUM_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        target_update_interval=args.target_update,
        buffer_capacity=args.buffer_size,
        batch_size=args.batch_size,
        length_weight=args.length_weight,
        report_quality_weight=args.report_quality_weight,
        device=args.device,
    )

    if args.resume_path and os.path.exists(args.resume_path):
        trainer.load(args.resume_path)
        logger.info(f"Resumed from checkpoint: {args.resume_path}")

    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / max(args.num_episodes, 1)

    best_reward = -float("inf")
    import time as _time
    _train_start = _time.time()

    for ep_idx in range(args.num_episodes):
        _ep_start = _time.time()
        task = tasks[ep_idx % len(tasks)]

        graph = QMIXGraph(
            llm_name=args.llm_name,
            agent_names=agent_names,
            execution_trace=True,
        )

        episode, _ = await run_episode(
            graph, trainer, task, epsilon,
            max_rounds=args.max_rounds, 
            length_goal=args.length_goal, 
            length_sigma=args.length_sigma,
        )

        total_tokens = episode.total_tokens

        total_reward = episode.total_reward
        score = Score.instance().current_score
        if score is None:
            score = 0

        trainer.replay_buffer.push(episode)

        train_info = trainer.train_step()

        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

        _ep_elapsed = _time.time() - _ep_start
        _total_elapsed = _time.time() - _train_start
        _avg_ep_time = _total_elapsed / (ep_idx + 1)
        _remaining = _avg_ep_time * (args.num_episodes - ep_idx - 1)

        task_preview = task[:80].replace("\n", " ")
        answer_preview = ReportState.instance().progress[:120].replace("\n", " ")
        loss_str = f"loss={train_info['loss']:.4f}" if train_info else "loss=n/a"

        print(f"\n--- Episode {ep_idx+1}/{args.num_episodes} "
              f"[{_ep_elapsed:.1f}s] ---")
        print(f"  Task:   {task_preview}...")
        print(f"  Output: {answer_preview}...")
        print(f"  score={score:.2f} | total reward={total_reward:.3f} | "
              f"tokens={total_tokens} | eps={epsilon:.3f} | {loss_str}")
        print(f"  Elapsed: {_total_elapsed:.0f}s | "
              f"ETA: {_remaining:.0f}s ({_remaining/60:.1f}min)")

        if total_reward > best_reward:
            best_reward = total_reward
            if args.save_path:
                trainer.save(args.save_path)

    if args.save_path:
        trainer.save(args.save_path)

    avg_r = trainer.replay_buffer.avg_reward
    avg_t = trainer.replay_buffer.avg_tokens
    print()
    print("=" * 60)
    print(f"  TRAINING COMPLETE: Redacting team")
    print("=" * 60)
    print(f"  Best Reward:  {best_reward:.4f}")
    print(f"  Avg Reward:   {avg_r:.4f}")
    print(f"  Avg Tokens:   {avg_t:.0f}")
    print(f"  Steps:        {trainer.training_step}")
    if args.save_path:
        print(f"  Model saved:  {args.save_path}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="QMIX Training for Multi-Agent Topology Optimization")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--llm_name", type=str, default="qwen3:8b")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--target_update", type=int, default=200)
    parser.add_argument("--length_weight", type=float, default=0.1)
    parser.add_argument("--report_quality_weight", type=float, default=1.0)
    parser.add_argument("--length_goal", type=int, default=25000)
    parser.add_argument("--length_sigma", type=int, default=8500)
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to an existing checkpoint to resume training from")
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    if args.save_path is None:
        os.makedirs("checkpoints", exist_ok=True)
        args.save_path = f"checkpoints/qmix_redacting_{datetime.now():%Y%m%d_%H%M}.pt"

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
