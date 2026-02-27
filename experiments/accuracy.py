"""Accuracy computation, result logging, and JSONL output."""

import json
import os
from typing import List, Dict
from datetime import datetime
from utils.log import get_logger

logger = get_logger("accuracy")


def compute_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Compute accuracy metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0, "partial_accuracy": 0.0,
            "total": 0, "correct": 0,
            "total_tokens": 0, "avg_tokens": 0,
            "total_cost": 0.0,
        }

    correct = sum(1 for r in results if r.get("score", 0) >= 0.99)
    partial = sum(r.get("score", 0) for r in results)
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_cost = sum(r.get("cost", 0) for r in results)

    return {
        "accuracy": correct / total,
        "partial_accuracy": partial / total,
        "total": total,
        "correct": correct,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total,
        "total_cost": total_cost,
    }


def save_results(results: List[Dict], output_path: str, metadata: Dict = None):
    """Save evaluation results to both JSON and JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    metrics = compute_accuracy(results)
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "metadata": metadata or {},
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    jsonl_path = output_path.replace(".json", ".jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _print_summary(metrics, output_path, metadata)
    return metrics


def _print_summary(metrics: Dict, output_path: str, metadata: Dict = None):
    """Pretty-print evaluation summary to terminal."""
    dataset = (metadata or {}).get("dataset", "unknown")
    llm = (metadata or {}).get("llm", "")
    n_agents = (metadata or {}).get("n_agents", "?")

    print()
    print("=" * 60)
    print(f"  RESULTS: {dataset.upper()}")
    print("=" * 60)
    print(f"  Model:       {llm}")
    print(f"  Agents:      {n_agents}")
    print(f"  Samples:     {metrics['total']}")
    print("-" * 60)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
    print(f"  Partial Acc: {metrics['partial_accuracy']:.4f}")
    print(f"  Avg Tokens:  {metrics['avg_tokens']:.0f}")
    print(f"  Total Tokens:{metrics['total_tokens']}")
    print(f"  Total Cost:  ${metrics['total_cost']:.4f}")
    print("-" * 60)
    print(f"  Saved to:    {output_path}")
    print(f"  JSONL:       {output_path.replace('.json', '.jsonl')}")
    print("=" * 60)
    print()
