"""
Entry point for running the handcrafted communication graph.

Mirrors the interface of the QMIX training runner so the two can be compared
under the same experimental harness.

Quickstart:
    import asyncio
    from handcrafted_graph.runner import run_handcrafted

    answers, tokens = asyncio.run(
        run_handcrafted(task="Write a technical report on graphene synthesis.")
    )
    print(answers[0])
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

from handcrafted_graph.graph import HandcraftedGraph
from handcrafted_graph.phases import PhaseConfig
from handcrafted_graph.scheduler import SkipStrategy
from handcrafted_graph.state import PhaseState
from utils.globals import ReportState, Score, LengthGoal, SourceBuffer
from utils.config import get_config
from utils.report_filter import filter_meta_commentary

logger = logging.getLogger("handcrafted_graph.runner")

# Default agent roster (matches the QMIX experiment config)
DEFAULT_AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]


async def run_handcrafted(
    task: str,
    llm_name: Optional[str] = None,
    agent_names: Optional[List[str]] = None,
    skip_strategy: SkipStrategy = SkipStrategy.ALWAYS_INCLUDE,
    execution_trace: bool = False,
    phases: Optional[List[PhaseConfig]] = None,
    max_tries: int = 3,
    max_time: int = 300,
    save_output: bool = True,
    export_pdf: bool = True,
) -> Tuple[List[str], int]:
    """Run the full handcrafted pipeline for a single report task.

    Args:
        task: The report subject / writing objective.
        llm_name: LLM model name (defaults to the value in configs/default.yaml).
        agent_names: Agent roster to use; defaults to DEFAULT_AGENTS.
        skip_strategy: How optional agents decide to skip rounds.
        execution_trace: Record a full execution trace for debugging.
        phases: Override the default phase sequence.
        max_tries: Retry attempts per agent on failure.
        max_time: Per-agent execution timeout in seconds.
        save_output: Write the raw markdown report to a fresh run folder under
            output/ (git-ignored). The folder path is logged.
        export_pdf: After saving, also convert the report to LaTeX and compile it
            to PDF in the same run folder. Requires save_output. Failures here are
            logged but do not abort the run (the raw markdown is always kept).

    Returns:
        (answers, total_tokens)
    """
    cfg = get_config()
    if llm_name is None:
        llm_name = cfg.get("llm", {}).get("default_model", "qwen3:8b")
    if agent_names is None:
        agent_names = DEFAULT_AGENTS

    _reset_singletons()

    graph = HandcraftedGraph(
        llm_name=llm_name,
        agent_names=agent_names,
        skip_strategy=skip_strategy,
        execution_trace=execution_trace,
        phases=phases,
    )

    logger.info(f"Starting handcrafted run | task='{task[:80]}...' | llm={llm_name}")

    answers, total_tokens = await graph.arun(
        input={"task": task},
        max_tries=max_tries,
        max_time=max_time,
    )

    # Strip pipeline-internal meta-commentary from the final output.
    # The execution trace is left unfiltered for debugging.
    answers = [filter_meta_commentary(a) for a in answers]

    logger.info(
        f"Run complete | tokens={total_tokens} | "
        f"report_length={len(answers[0])} chars"
    )

    if save_output and answers:
        from utils.report_export import save_raw_report
        run_dir = save_raw_report(task=task, report=answers[0])
        logger.info(f"Raw report saved to {run_dir}")

        pdf_path = None
        if export_pdf:
            try:
                from utils.markdown_to_latex import convert_run_dir
                from utils.compile_pdf import compile_run_dir
                convert_run_dir(run_dir)
                pdf_path = compile_run_dir(run_dir)
                logger.info(f"Report PDF compiled to {pdf_path}")
            except Exception as e:
                logger.warning(
                    f"PDF export failed ({e}). Raw markdown is available at {run_dir}."
                )

        # Surface the output location(s) to the user (the report body itself is
        # written to disk, not printed to the terminal).
        print(f"\nReport saved to: {run_dir}")
        if pdf_path is not None:
            print(f"PDF: {pdf_path}")

    if execution_trace:
        from utils.globals import ExecutionTrace
        ExecutionTrace.instance().save_trace("handcrafted_trace.json")

    return answers, total_tokens


def _reset_singletons() -> None:
    """Reset all global state before a new run."""
    ReportState.instance().reset()
    PhaseState.instance().reset()
    try:
        Score.instance().reset()
        LengthGoal.instance().reset()
    except Exception:
        pass
    SourceBuffer.instance().reset()


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the handcrafted graph pipeline.")
    parser.add_argument("task", help="Report writing task / subject.")
    parser.add_argument("--llm", default=None, help="LLM model name.")
    parser.add_argument(
        "--skip-strategy",
        choices=[s.value for s in SkipStrategy],
        default=SkipStrategy.ALWAYS_INCLUDE.value,
        help="Skip strategy for optional agents.",
    )
    parser.add_argument("--trace", action="store_true", help="Save execution trace.")
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip LaTeX/PDF export (only save the raw markdown report).",
    )
    args = parser.parse_args()

    answers, tokens = asyncio.run(
        run_handcrafted(
            task=args.task,
            llm_name=args.llm,
            skip_strategy=SkipStrategy(args.skip_strategy),
            execution_trace=args.trace,
            export_pdf=not args.no_pdf,
        )
    )

    print("\n" + "=" * 60)
    print(f"Total tokens used: {tokens}")
    print("=" * 60)
