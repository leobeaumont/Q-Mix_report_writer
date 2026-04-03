"""
Query the QMIX multi-agent system with a plain text question.

Usage:
    python query_qmix.py "What is the derivative of x^2?" --domain math
    python query_qmix.py "Write a function to reverse a string" --domain coding
    python query_qmix.py "Explain the causes of WWI" --domain reasoning
    python query_qmix.py "What is the capital of France?" --domain reasoning --model_path checkpoints_ollama/qmix_unified.pt

Domains:
    reasoning  ->  General knowledge, analysis, open questions  (uses mmlu_pro prompt set)
    math       ->  Mathematical problems, calculations          (uses aime prompt set)
    coding     ->  Programming tasks, algorithms                (uses humaneval prompt set)
"""

import os
import sys
import asyncio
import argparse
import torch

# Make sure the project root is on the path regardless of where the script is called from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from graph.graph import QMIXGraph
from qmix.qmix_trainer import QMIXTrainer
from utils.config import get_config

# ── Domain mapping ────────────────────────────────────────────────────────────
# Maps user-friendly names to the keys registered in PromptSetRegistry
# and the agent configuration keys in default.yaml
DOMAIN_MAP = {
    "reasoning": {
        "prompt_key": "mmlu_pro",   # AgenticPromptSet — most general purpose
        "config_key": "reasoning",
    },
    "math": {
        "prompt_key": "aime",       # MathPromptSet
        "config_key": "math",
    },
    "coding": {
        "prompt_key": "humaneval",  # CodingPromptSet
        "config_key": "coding",
    },
}

# ── Dimension helpers ─────────────────────────────────────────────────────────
# Derived directly from QMIXGraph.get_observation_features() in graph/graph.py:
#   task_features (32) + agent_identity one-hot (16) + [has_output, n_neighbors, token_usage] (3) = 51
OBS_DIM_PER_AGENT = 51

def get_obs_dim() -> int:
    return OBS_DIM_PER_AGENT

def get_state_dim(n_agents: int) -> int:
    # Derived from QMIXGraph.get_global_state(): obs.flatten() + graph_stats (3)
    return n_agents * OBS_DIM_PER_AGENT + 3


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Query the QMIX multi-agent system with a local Ollama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        help="The question or task for the agents to solve.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="reasoning",
        choices=list(DOMAIN_MAP.keys()),
        help="The type of task. Controls which agents and prompts are used. (default: reasoning)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to a trained QMIX checkpoint (.pt file). Runs with untrained weights if omitted.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of communication rounds between agents. (default: 2)",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = get_config()
    llm_name = config.get("llm", {}).get("default_model", "tinyllama")
    domain_info = DOMAIN_MAP[args.domain]
    prompt_key = domain_info["prompt_key"]
    config_key = domain_info["config_key"]

    agent_names = (
        config.get("agent_configs", {})
              .get(config_key, {})
              .get("agents", ["ReasoningAgent", "ReasoningAgent", "AnalyzeAgent"])
    )
    n_agents = len(agent_names)

    print()
    print("=" * 56)
    print("  QMIX Multi-Agent System")
    print("=" * 56)
    print(f"  Model    : {llm_name}")
    print(f"  Domain   : {args.domain} -> prompt set: {prompt_key}")
    print(f"  Agents   : {agent_names}")
    print(f"  Rounds   : {args.rounds}")
    print("=" * 56)
    print()

    # ── QMIX trainer (the 'brain' that decides agent communication) ───────────
    obs_dim = get_obs_dim()
    state_dim = get_state_dim(n_agents)

    trainer = QMIXTrainer(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        device="cpu",
    )

    if args.model_path and os.path.exists(args.model_path):
        trainer.load(args.model_path)
        print(f"[*] Loaded trained checkpoint from: {args.model_path}")
    else:
        if args.model_path:
            print(f"[!] Checkpoint not found at '{args.model_path}'.")
        print("[!] Running with UNTRAINED weights — communication topology is random.")
        print()

    # ── Graph (the 'body' that runs the agents) ───────────────────────────────
    # prompt_key must match a key registered in PromptSetRegistry
    graph = QMIXGraph(
        domain=prompt_key,
        llm_name=llm_name,
        agent_names=agent_names,
    )

    # ── Action selection ──────────────────────────────────────────────────────
    obs = torch.tensor(
        graph.get_observation_features(args.query), dtype=torch.float32
    )
    adj = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32)
    hidden = trainer.agent_network.init_hidden(n_agents)

    with torch.no_grad():
        actions, _ = trainer.select_actions(obs, adj, hidden, epsilon=0.0)

    action_names = ["solo", "broadcast", "selective", "aggregate", "execute_verify", "debate"]
    action_labels = [action_names[a] for a in actions.tolist()]
    print(f"[*] Agent communication actions: {action_labels}")
    print()
    print(f"[*] Running {n_agents} agents over {args.rounds} round(s)...")
    print()

    # ── Execute ───────────────────────────────────────────────────────────────
    answers, tokens = await graph.arun(
        {"task": args.query},
        num_rounds=args.rounds,
        actions=actions,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    print("=" * 56)
    print("  FINAL ANSWER")
    print("=" * 56)
    print(answers[0] if answers else "No response produced.")
    print("=" * 56)
    print(f"  Total tokens used: {tokens}")
    print()


if __name__ == "__main__":
    asyncio.run(main())