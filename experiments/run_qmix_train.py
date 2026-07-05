"""
QMIX training loop — CLI entry point (upgrade-plan Stage 4.6).

Each episode runs the handcrafted phase pipeline in training shape (decision
D6: PLANNING→RESEARCH→DRAFTING, no correction phases, no finalization) with a
QMIXRoundController choosing the communication topology; recorded transitions
feed the replay buffer and a train_step() runs per episode.

The task list (datasets/tasks.py) is wired here — the report evaluator
(grounded judges + reward composition) lives in the package
(qmix_report_writer/evaluation) and is built by the runner.

Usage (from project root):
    python experiments/run_qmix_train.py --num-episodes 50 --trace
    python experiments/run_qmix_train.py --resume checkpoints/qmix_v2.pt
"""

import os
import sys
import warnings
import argparse
import asyncio
from datetime import datetime

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tasks import tasks
from qmix_report_writer.qmix.runner import run_qmix_train
from qmix_report_writer.utils.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Train the QMIX topology policy.")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM model name (defaults to configs/default.yaml).")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Episodes to run (defaults to qmix.training.num_episodes).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Checkpoint path (default: checkpoints/qmix_v2_<ts>.pt).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Existing checkpoint to resume from.")
    parser.add_argument("--trace", action="store_true",
                        help="Save each episode's execution trace to qmix_trace.json.")
    parser.add_argument("--max-tries", type=int, default=3,
                        help="Retry attempts per agent on LLM failure.")
    parser.add_argument("--max-time", type=int, default=300,
                        help="Per-agent execution timeout in seconds.")
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None:
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/qmix_v2_{datetime.now():%Y%m%d_%H%M}.pt"

    cfg = get_config()
    llm_name = args.llm or cfg.get("llm", {}).get("default_model", "qwen3:8b")

    print()
    print("=" * 60)
    print("  QMIX TRAINING: report-writing episodes (D6 shape)")
    print("=" * 60)
    print(f"  LLM:        {llm_name}")
    print(f"  Episodes:   {args.num_episodes or cfg.get('qmix', {}).get('training', {}).get('num_episodes', 500)}")
    print(f"  Device:     {args.device}")
    print(f"  Checkpoint: {save_path}")
    print("=" * 60)
    print()

    asyncio.run(
        run_qmix_train(
            tasks=tasks,
            llm_name=llm_name,
            num_episodes=args.num_episodes,
            device=args.device,
            save_path=save_path,
            resume_path=args.resume,
            execution_trace=args.trace,
            max_tries=args.max_tries,
            max_time=args.max_time,
        )
    )


if __name__ == "__main__":
    main()
