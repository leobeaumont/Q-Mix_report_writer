"""
QMIX Training Loop for Multi-Agent Communication Topology Optimization.

This script implements the centralized training phase:
1. Initialize QMIX trainer with agent networks and mixing network
2. For each episode:
   a. Sample a task from the dataset
   b. Agents select actions via epsilon-greedy using their Q-networks
   c. Execute the multi-agent graph with QMIX-selected topology
   d. Compute reward: accuracy * w_acc - token_usage * w_token
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix.qmix_trainer import QMIXTrainer
from qmix.replay_buffer import Episode, EpisodeStep
from qmix.agent_network import NUM_ACTIONS
from graph.graph import QMIXGraph
from datasets import get_dataset
from utils.log import get_logger
from utils.globals import PromptTokens, CompletionTokens, Cost

logger = get_logger("qmix_train")

DOMAIN_MAP = {
    "humaneval": "humaneval",
    "livecodebench": "livecodebench",
    "livecodebench_testgen": "livecodebench",
    "mmlu_pro": "mmlu_pro",
    "gaia": "gaia",
    "frontierscience": "gaia",
    "aime_2024": "aime",
    "aime_2025": "aime",
    "aime_2026": "aime",
    "beyond_aime": "aime",
    "hmmt_2025": "hmmt",
}

AGENT_CONFIGS = {
    "humaneval": ["CodeWriter", "CodeWriter", "AnalyzeAgent"],
    "livecodebench": ["CodeWriter", "CodeWriter", "AnalyzeAgent"],
    "livecodebench_testgen": ["CodeWriter", "CodeWriter", "AnalyzeAgent"],
    "mmlu_pro": ["ReasoningAgent", "ReasoningAgent", "AnalyzeAgent"],
    "gaia": ["ReasoningAgent", "ReasoningAgent", "AnalyzeAgent"],
    "frontierscience": ["ReasoningAgent", "ReasoningAgent", "AnalyzeAgent"],
    "aime_2024": ["MathSolver", "MathSolver", "AnalyzeAgent"],
    "aime_2025": ["MathSolver", "MathSolver", "AnalyzeAgent"],
    "aime_2026": ["MathSolver", "MathSolver", "AnalyzeAgent"],
    "beyond_aime": ["MathSolver", "MathSolver", "AnalyzeAgent"],
    "hmmt_2025": ["MathSolver", "MathSolver", "AnalyzeAgent"],
}


def get_obs_dim(n_agents, task_feature_dim=32):
    """Calculate observation dimension per agent."""
    agent_id_dim = 16
    extra_features = 3  # has_output, n_neighbors_ratio, token_ratio
    return task_feature_dim + agent_id_dim + extra_features


def get_state_dim(n_agents, obs_dim):
    """Calculate global state dimension for mixing network."""
    graph_stats = 3  # total_edges, density, n_agents
    return n_agents * obs_dim + graph_stats


async def run_episode(
    graph: QMIXGraph,
    trainer: QMIXTrainer,
    task_text: str,
    epsilon: float,
    num_rounds: int = 2,
) -> Episode:
    """Run a single episode of QMIX-driven multi-agent interaction."""
    episode = Episode()
    n_agents = graph.n_agents

    hidden = trainer.agent_network.init_hidden(n_agents)
    obs = torch.tensor(graph.get_observation_features(task_text), dtype=torch.float32)
    adj = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32)

    for step in range(num_rounds):
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
            done=(step == num_rounds - 1),
            token_usage=tokens_used,
        )
        episode.add_step(step_data)

        obs = new_obs
        adj = new_adj

    return episode, answers


async def train(args):
    """Main training loop."""
    dataset_name = args.dataset
    domain = DOMAIN_MAP.get(dataset_name, dataset_name)
    agent_names = AGENT_CONFIGS.get(dataset_name, ["ReasoningAgent"] * 3)
    n_agents = len(agent_names)

    print()
    print("=" * 60)
    print(f"  QMIX TRAINING: {dataset_name.upper()}")
    print("=" * 60)
    print(f"  LLM:       {args.llm_name}")
    print(f"  Agents:    {agent_names}")
    print(f"  Episodes:  {args.num_episodes}")
    print(f"  Rounds:    {args.num_rounds}")
    print(f"  Device:    {args.device}")
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
        token_penalty_weight=args.token_penalty,
        accuracy_reward_weight=args.accuracy_weight,
        device=args.device,
    )

    dataset = get_dataset(dataset_name, split=args.split, limit=args.data_limit, data_path=args.data_path)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    if len(dataset) == 0:
        logger.error(f"No samples loaded for {dataset_name}. Check that HuggingFace 'datasets' library "
                      f"is installed (`pip install datasets`) and can access the internet, or provide "
                      f"--data_path to a local JSONL file.")
        return

    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / max(args.num_episodes, 1)

    best_reward = -float("inf")
    import time as _time
    _train_start = _time.time()

    for ep_idx in range(args.num_episodes):
        _ep_start = _time.time()
        sample = dataset[ep_idx % len(dataset)]

        graph = QMIXGraph(
            domain=domain,
            llm_name=args.llm_name,
            agent_names=agent_names,
            decision_method=args.decision_method,
        )

        episode, answers = await run_episode(
            graph, trainer, sample.task, epsilon, num_rounds=args.num_rounds,
        )

        answer_text = answers[0] if answers else ""
        accuracy = dataset.evaluate(answer_text, sample.ground_truth)
        total_tokens = episode.total_tokens

        reward = trainer.compute_reward(accuracy, total_tokens, max_tokens=args.max_tokens)

        for step in episode.steps:
            step.team_reward = reward

        trainer.replay_buffer.push(episode)

        train_info = trainer.train_step()

        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

        _ep_elapsed = _time.time() - _ep_start
        _total_elapsed = _time.time() - _train_start
        _avg_ep_time = _total_elapsed / (ep_idx + 1)
        _remaining = _avg_ep_time * (args.num_episodes - ep_idx - 1)

        problem_idx = ep_idx % len(dataset)
        task_preview = sample.task[:80].replace("\n", " ")
        answer_preview = answer_text[:120].replace("\n", " ")
        loss_str = f"loss={train_info['loss']:.4f}" if train_info else "loss=n/a"

        print(f"\n--- Episode {ep_idx+1}/{args.num_episodes} "
              f"[problem #{problem_idx}, {_ep_elapsed:.1f}s] ---")
        print(f"  Task:   {task_preview}...")
        print(f"  Output: {answer_preview}...")
        print(f"  acc={accuracy:.2f} | reward={reward:.3f} | "
              f"tokens={total_tokens} | eps={epsilon:.3f} | {loss_str}")
        print(f"  Elapsed: {_total_elapsed:.0f}s | "
              f"ETA: {_remaining:.0f}s ({_remaining/60:.1f}min)")

        if reward > best_reward:
            best_reward = reward
            if args.save_path:
                trainer.save(args.save_path)

    if args.save_path:
        trainer.save(args.save_path)

    avg_r = trainer.replay_buffer.avg_reward
    avg_t = trainer.replay_buffer.avg_tokens
    print()
    print("=" * 60)
    print(f"  TRAINING COMPLETE: {dataset_name.upper()}")
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
    parser.add_argument("--dataset", type=str, required=True, choices=list(DOMAIN_MAP.keys()))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--target_update", type=int, default=200)
    parser.add_argument("--token_penalty", type=float, default=0.1)
    parser.add_argument("--accuracy_weight", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    if args.save_path is None:
        os.makedirs("checkpoints", exist_ok=True)
        args.save_path = f"checkpoints/qmix_{args.dataset}_{datetime.now():%Y%m%d_%H%M}.pt"

    asyncio.run(train(args))


if __name__ == "__main__":
    main()
