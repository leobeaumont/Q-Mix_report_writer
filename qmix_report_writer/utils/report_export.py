"""
Report output management.

Every pipeline run that produces a report gets its own timestamped folder under
the (git-ignored) ``output/`` directory at the project root. That folder is the
single home for all artifacts derived from the run:

    output/
      2026-06-17_153012_graphene-synthesis/
        report_raw.md      <- raw markdown straight out of the pipeline
        report.tex         <- LaTeX conversion        (added in a later step)
        report.pdf         <- compiled PDF            (added in a later step)

This module currently handles step 1: creating the run folder and dropping the
raw markdown report into it. The LaTeX conversion and PDF compilation hook into
the same folder via :func:`report_dir` / the returned path.

Usage:
    from qmix_report_writer.utils.report_export import save_raw_report

    run_dir = save_raw_report(task="Write a report on graphene synthesis.",
                              report=markdown_text)
    # -> Path(".../output/2026-06-17_153012_graphene-synthesis")
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_output_dir

# Standard artifact filenames within a run folder.
RAW_MARKDOWN_NAME = "report_raw.md"
LATEX_NAME = "report.tex"
PDF_NAME = "report.pdf"

# Number of characters of the query kept in the folder slug.
_MAX_SLUG_LEN = 50


def _slugify(text: str) -> str:
    """Turn an arbitrary query string into a filesystem-safe, readable slug.

    Lowercases, replaces any run of non-alphanumeric characters with a single
    hyphen, trims to a sane length, and strips leading/trailing hyphens. Falls
    back to ``"report"`` when nothing usable remains (e.g. an all-symbol query).
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if len(slug) > _MAX_SLUG_LEN:
        # Cut at the last hyphen before the limit so words stay whole.
        slug = slug[:_MAX_SLUG_LEN].rsplit("-", 1)[0]
    return slug or "report"


def create_run_dir(task: str, base_dir: Optional[Path] = None) -> Path:
    """Create and return a fresh output folder for a single report run.

    The folder name is ``<timestamp>_<query-slug>`` so runs sort chronologically
    and stay human-identifiable. If a folder of that exact name already exists
    (same second, same query), a numeric suffix is appended to avoid clobbering.

    Args:
        task: The original query / writing objective given to the report writer.
        base_dir: Override for the parent directory (defaults to ``output/``).
                  Mainly useful for tests.

    Returns:
        Path to the newly created run directory.
    """
    root = Path(base_dir) if base_dir is not None else get_output_dir()
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_name = f"{timestamp}_{_slugify(task)}"

    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def save_raw_report(
    task: str,
    report: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Create a run folder and write the raw markdown report into it.

    Args:
        task: The original query given to the report writer (used to name the
              folder and recorded as a comment at the top of the markdown file).
        report: The raw markdown report text produced by the pipeline.
        base_dir: Override for the parent directory (defaults to ``output/``).

    Returns:
        Path to the run directory containing ``report_raw.md``.
    """
    run_dir = create_run_dir(task, base_dir=base_dir)
    raw_path = run_dir / RAW_MARKDOWN_NAME

    # Prepend the original query as an HTML comment so the source folder is
    # self-describing without altering the rendered markdown.
    header = f"<!-- Query: {task} -->\n<!-- Generated: {datetime.now().isoformat(timespec='seconds')} -->\n\n"
    raw_path.write_text(header + report, encoding="utf-8")

    return run_dir
