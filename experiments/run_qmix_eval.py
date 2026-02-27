"""
QMIX Evaluation: Decentralized Execution with Trained Policy.

Once trained, agents are deployed for fully decentralized execution.
Each agent makes optimal decisions based on its local, neighbor-informed perspective.

Evaluation flow:
1. Load trained QMIX model
2. For each test sample:
   a. Agents select greedy actions using their Q-networks (no exploration)
   b. Execute multi-agent graph with learned topology
   c. Record accuracy and token usage
3. Report final metrics
"""

import os
import sys
import argparse
import asyncio
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix.qmix_trainer import QMIXTrainer
from qmix.agent_network import NUM_ACTIONS
from graph.graph import QMIXGraph, TOPOLOGY_PRESETS
from datasets import get_dataset
from experiments.accuracy import compute_accuracy, save_results
from experiments.run_qmix_train import (
    DOMAIN_MAP, AGENT_CONFIGS, get_obs_dim, get_state_dim,
)
from utils.log import get_logger
from utils.globals import PromptTokens, CompletionTokens, Cost

logger = get_logger("qmix_eval")


async def evaluate_qmix(args):
    """Evaluate with trained QMIX policy (decentralized execution)."""
    dataset_name = args.dataset
    domain = DOMAIN_MAP.get(dataset_name, dataset_name)
    agent_names = AGENT_CONFIGS.get(dataset_name, ["ReasoningAgent"] * 3)
    n_agents = len(agent_names)

    obs_dim = get_obs_dim(n_agents)
    state_dim = get_state_dim(n_agents, obs_dim)

    trainer = QMIXTrainer(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=NUM_ACTIONS,
        device=args.device,
    )

    if args.model_path and os.path.exists(args.model_path):
        trainer.load(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.warning("No model loaded - using random initialization")

    dataset = get_dataset(dataset_name, split=args.split, limit=args.limit, data_path=args.data_path)
    logger.info(f"Evaluating {dataset_name} ({args.split}): {len(dataset)} samples, {n_agents} agents")

    if len(dataset) == 0:
        logger.error(f"No samples for {dataset_name}. Provide --data_path or check network.")
        return save_results([], args.output_path, metadata={"dataset": dataset_name, "llm": args.llm_name})

    results = []
    import time as _time
    _eval_start = _time.time()
    total_samples = len(dataset)

    for idx, sample in enumerate(dataset.samples):
        _s_start = _time.time()
        Cost.instance().reset()
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()

        graph = QMIXGraph(
            domain=domain,
            llm_name=args.llm_name,
            agent_names=agent_names,
            decision_method=args.decision_method,
        )

        hidden = trainer.agent_network.init_hidden(n_agents)
        obs = torch.tensor(graph.get_observation_features(sample.task), dtype=torch.float32)
        adj = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32)

        actions, _ = trainer.select_actions(obs, adj, hidden, epsilon=0.0)

        try:
            answers, tokens_used = await graph.arun(
                {"task": sample.task},
                num_rounds=args.num_rounds,
                actions=actions,
            )
            answer_text = answers[0] if answers else ""
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            answer_text = ""
            tokens_used = 0

        score = dataset.evaluate(answer_text, sample.ground_truth)

        results.append({
            "task_id": sample.task_id,
            "score": score,
            "tokens_used": tokens_used,
            "prediction": answer_text[:500],
            "ground_truth": sample.ground_truth[:200],
            "actions": actions.tolist(),
            "cost": Cost.instance().value,
        })

        _s_elapsed = _time.time() - _s_start
        _total_elapsed = _time.time() - _eval_start
        _avg_time = _total_elapsed / (idx + 1)
        _remaining = _avg_time * (total_samples - idx - 1)
        running = compute_accuracy(results)

        task_preview = sample.task[:80].replace("\n", " ")
        ans_preview = answer_text[:120].replace("\n", " ")
        gt_preview = sample.ground_truth[:60].replace("\n", " ")

        print(f"\n--- [{idx+1}/{total_samples}] problem #{idx} ({_s_elapsed:.1f}s) ---")
        print(f"  Task:   {task_preview}...")
        print(f"  Output: {ans_preview}...")
        print(f"  GT:     {gt_preview}")
        print(f"  score={score:.2f} | tokens={tokens_used} | "
              f"actions={actions.tolist()}")
        print(f"  Running acc={running['accuracy']:.3f} | "
              f"avg_tok={running['avg_tokens']:.0f} | "
              f"Elapsed: {_total_elapsed:.0f}s | "
              f"ETA: {_remaining:.0f}s ({_remaining/60:.1f}min)")

    metrics = save_results(
        results,
        args.output_path,
        metadata={
            "dataset": dataset_name,
            "llm": args.llm_name,
            "model_path": args.model_path,
            "n_agents": n_agents,
            "agent_names": agent_names,
            "num_rounds": args.num_rounds,
        },
    )
    return metrics


async def evaluate_baseline(args):
    """Evaluate with a fixed topology (baseline comparison)."""
    dataset_name = args.dataset
    domain = DOMAIN_MAP.get(dataset_name, dataset_name)
    agent_names = AGENT_CONFIGS.get(dataset_name, ["ReasoningAgent"] * 3)

    dataset = get_dataset(dataset_name, split=args.split, limit=args.limit, data_path=args.data_path)
    logger.info(f"Baseline eval ({args.topology}) {dataset_name} ({args.split}): {len(dataset)} samples")

    if len(dataset) == 0:
        logger.error(f"No samples for {dataset_name}. Provide --data_path or check network.")
        output_path = args.output_path.replace(".json", f"_{args.topology}.json")
        return save_results([], output_path, metadata={"dataset": dataset_name, "topology": args.topology})

    results = []

    for idx, sample in enumerate(tqdm(dataset.samples, desc=f"Baseline {args.topology}")):
        Cost.instance().reset()
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()

        graph = QMIXGraph(
            domain=domain,
            llm_name=args.llm_name,
            agent_names=agent_names,
            decision_method=args.decision_method,
            fixed_topology=args.topology,
        )

        try:
            answers, tokens_used = await graph.arun(
                {"task": sample.task},
                num_rounds=args.num_rounds,
            )
            answer_text = answers[0] if answers else ""
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            answer_text = ""
            tokens_used = 0

        score = dataset.evaluate(answer_text, sample.ground_truth)
        results.append({
            "task_id": sample.task_id,
            "score": score,
            "tokens_used": tokens_used,
            "prediction": answer_text[:500],
            "ground_truth": sample.ground_truth[:200],
            "topology": args.topology,
        })

    output_path = args.output_path.replace(".json", f"_{args.topology}.json")
    metrics = save_results(results, output_path, metadata={
        "dataset": dataset_name, "topology": args.topology,
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="QMIX Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DOMAIN_MAP.keys()))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--mode", type=str, default="qmix", choices=["qmix", "baseline"])
    parser.add_argument("--topology", type=str, default="full",
                        choices=list(TOPOLOGY_PRESETS.keys()))
    args = parser.parse_args()

    if args.output_path is None:
        os.makedirs("result", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_path = f"result/{args.dataset}_{args.mode}_{ts}.json"

    if args.mode == "qmix":
        asyncio.run(evaluate_qmix(args))
    else:
        asyncio.run(evaluate_baseline(args))


if __name__ == "__main__":
    main()
