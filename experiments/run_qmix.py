"""
QMIX pipeline — single-task inference runner (upgrade-plan Stage 4.6).

Generates one report with the (trained) QMIX policy driving the communication
topology inside the handcrafted phase pipeline. Full phases, finalization on —
the same artifacts as run_handcrafted.

Usage (from project root):
    python experiments/run_qmix.py --task "Write a report on graphene synthesis." --model-path checkpoints/qmix.pt
    python experiments/run_qmix.py --task-index 0 --trace --no-pdf
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
from qmix_report_writer.handcrafted_graph.graph import NoCorpusCoverageError
from qmix_report_writer.qmix.runner import run_qmix
from qmix_report_writer.utils.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Run the QMIX-controlled pipeline (inference).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", type=str, default=None,
                       help="Report task string to use.")
    group.add_argument("--task-index", type=int, default=0,
                       help="Index into the built-in tasks list (default: 0).")
    parser.add_argument("--model-path", type=str, default=None,
                        help="QMIX checkpoint to load (untrained policy if omitted).")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM model name (defaults to configs/default.yaml).")
    parser.add_argument("--trace", action="store_true",
                        help="Save execution trace to qmix_trace.json.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip LaTeX/PDF export (only save the raw markdown report).")
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
    print("  QMIX PIPELINE RUN (inference)")
    print("=" * 60)
    print(f"  LLM:        {llm_name}")
    print(f"  Checkpoint: {args.model_path or '(none — untrained policy)'}")
    print(f"  Trace:      {args.trace}")
    print(f"  Task:       {task[:80]}...")
    print("=" * 60)
    print()

    try:
        answers, total_tokens = asyncio.run(
            run_qmix(
                task=task,
                model_path=args.model_path,
                llm_name=llm_name,
                execution_trace=args.trace,
                max_tries=args.max_tries,
                max_time=args.max_time,
                export_pdf=not args.no_pdf,
            )
        )
    except NoCorpusCoverageError as exc:
        print()
        print("=" * 60)
        print("  ABORTED — no corpus coverage for this task")
        print("=" * 60)
        print(f"  {exc}")
        print("=" * 60)
        sys.exit(2)

    print()
    print("=" * 60)
    print(f"  DONE — tokens used: {total_tokens}")
    print(f"  Report length:      {len(answers[0])} chars")
    print("=" * 60)


if __name__ == "__main__":
    main()
