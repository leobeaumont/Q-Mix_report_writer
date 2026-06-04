"""
Handcrafted Graph — single-task runner and smoke-test entry point.

Run one report task through the phase-based pipeline and optionally save the
execution trace for visualization with utils/visualization.py.

Usage (from project root):
    python experiments/run_handcrafted.py --task "Write a report on graphene synthesis."
    python experiments/run_handcrafted.py --task-index 0 --trace --llm qwen3:8b

The --trace flag writes handcrafted_trace.json in the project root, which can
be opened with:
    python utils/visualization.py   (after pointing load_trace() at the file)
"""

import os
import sys
import warnings
import argparse
import asyncio

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tasks import tasks
from handcrafted_graph.runner import run_handcrafted
from handcrafted_graph.scheduler import SkipStrategy
from utils.globals import ExecutionTrace, ReportState
from utils.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Run the handcrafted graph pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", type=str, default=None,
                       help="Report task string to use.")
    group.add_argument("--task-index", type=int, default=0,
                       help="Index into the built-in tasks list (default: 0).")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM model name (defaults to configs/default.yaml).")
    parser.add_argument("--trace", action="store_true",
                        help="Save execution trace to handcrafted_trace.json.")
    parser.add_argument("--skip-strategy",
                        choices=[s.value for s in SkipStrategy],
                        default=SkipStrategy.TEMPORAL_HEURISTIC.value,
                        help="Skip strategy for optional agents.")
    parser.add_argument("--max-tries", type=int, default=3,
                        help="Retry attempts per agent on LLM failure.")
    parser.add_argument("--max-time", type=int, default=300,
                        help="Per-agent execution timeout in seconds.")
    args = parser.parse_args()

    task = args.task if args.task else tasks[args.task_index % len(tasks)]

    cfg = get_config()
    llm_name = args.llm or cfg.get("llm", {}).get("default_model", "qwen3:8b")

    print()
    print("=" * 60)
    print("  HANDCRAFTED GRAPH RUN")
    print("=" * 60)
    print(f"  LLM:           {llm_name}")
    print(f"  Skip strategy: {args.skip_strategy}")
    print(f"  Trace:         {args.trace}")
    print(f"  Task:          {task[:80]}...")
    print("=" * 60)
    print()

    answers, total_tokens = asyncio.run(
        run_handcrafted(
            task=task,
            llm_name=llm_name,
            skip_strategy=SkipStrategy(args.skip_strategy),
            execution_trace=args.trace,
            max_tries=args.max_tries,
            max_time=args.max_time,
        )
    )

    report = answers[0]
    progress = ReportState.instance().progress

    print()
    print("=" * 60)
    print(f"  DONE — tokens used: {total_tokens}")
    print(f"  Report length:      {len(report)} chars")
    print("=" * 60)
    print()
    print("--- Report ---")
    print(report)
    print()

    if args.trace:
        trace_path = "handcrafted_trace.json"
        ExecutionTrace.instance().save_trace(trace_path)
        print(f"Execution trace saved to: {trace_path}")
        print("Visualize with:  python utils/visualization.py")


if __name__ == "__main__":
    main()
