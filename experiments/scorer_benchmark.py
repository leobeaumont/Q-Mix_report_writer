"""
Scorer benchmark harness (training_eval_upgrade_plan.md Stage 0.2, report A2.2).

Measures the report evaluator — BEFORE and AFTER the Part-A rework — on the
versioned real papers in tests/test_documents/:

  * ranking accuracy — over all ordered same-paper version pairs, the fraction
    where the newer (better) version scores higher;
  * repeat variance  — score the same document k times, per-run σ (the
    reward's noise floor);
  * corruption probes — off-topic section injected / sections shuffled /
    numbers perturbed: the score must drop vs the clean run.

The scoring path is a pluggable async *adapter* `(task, chunks) -> result
dict`, so the same protocol runs the legacy `experiments.eval.report_score`
(Stage 0.3 baseline) and the Stage-2 evaluator (2.8 re-run).

Metric functions are pure and offline-tested (test_training_eval_acceptance
Stage 0.2); actually scoring documents needs the live Ollama judges.

Usage (from the repo root, Ollama serving the judge model):
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py rank
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py repeat --doc "Towards(v22)" -k 5
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py corrupt --doc "Towards(v22)"
Results land in tests/scoring_results/benchmark_<mode>_<timestamp>.{json,md}.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime
from statistics import pstdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCS_DIR = os.path.join("tests", "test_documents")
RESULTS_DIR = os.path.join("tests", "scoring_results")
CHUNK_SIZE = 4000

_VERSION_RE = re.compile(r"v(\d+)", re.IGNORECASE)

_OFFTOPIC_SECTION = (
    "\n\n# A Note on Sourdough Baking\n\n"
    "Maintaining a healthy sourdough starter requires feeding it twice daily "
    "with equal parts flour and water. The ideal proofing temperature is "
    "around 26 °C, and a well-developed crumb depends on gentle stretch-and-fold "
    "cycles rather than aggressive kneading. None of this has any relation to "
    "the subject of this document.\n\n"
)


# ---------------------------------------------------------------------------
# Metrics (pure, offline-tested)
# ---------------------------------------------------------------------------

def ranking_accuracy(pairs):
    """Fraction of (older_score, newer_score) pairs where the newer wins.

    Ties count as half a win.
    """
    if not pairs:
        return 0.0
    wins = sum(1.0 if new > old else (0.5 if new == old else 0.0)
               for old, new in pairs)
    return wins / len(pairs)


def repeat_variance(scores):
    """Population standard deviation of repeated scores of the SAME input."""
    if len(scores) < 2:
        return 0.0
    return pstdev(scores)


# ---------------------------------------------------------------------------
# Corruption probes (deterministic, pure)
# ---------------------------------------------------------------------------

def inject_offtopic_section(doc: str) -> str:
    """Insert a blatantly off-topic section in the middle of the document."""
    mid = len(doc) // 2
    split = doc.find("\n\n", mid)
    if split == -1:
        split = mid
    return doc[:split] + _OFFTOPIC_SECTION + doc[split:]


def shuffle_sections(doc: str) -> str:
    """Deterministically reorder the document's parts (breaks narrative flow)."""
    parts = re.split(r"(?m)^(?=#)", doc)
    parts = [p for p in parts if p.strip()]
    if len(parts) < 2:  # no headings — fall back to paragraph blocks
        parts = [p for p in doc.split("\n\n") if p.strip()]
    if len(parts) < 2:
        return doc + "\n\n" + doc[: len(doc) // 2]  # degenerate input, still altered
    rng = random.Random(42)
    shuffled = parts[:]
    while shuffled == parts:
        rng.shuffle(shuffled)
    return "\n\n".join(p.strip() for p in shuffled)


def perturb_numbers(doc: str) -> str:
    """Corrupt every number in the document (breaks factual precision)."""

    def _twist(match):
        text = match.group(0)
        digits = "".join("9" if d in "01234" else "1" for d in text if d.isdigit())
        return digits or text

    corrupted = re.sub(r"\d+(?:\.\d+)?", _twist, doc)
    if corrupted == doc:  # no numbers at all — inject a wrong one
        corrupted = doc + "\n\nAll measurements above equal 999999 units."
    return corrupted


CORRUPTIONS = {
    "offtopic": inject_offtopic_section,
    "shuffled": shuffle_sections,
    "numbers": perturb_numbers,
}


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    from pypdf import PdfReader  # local import: only the live paths need it
    reader = PdfReader(pdf_path)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _version_key(stem: str):
    """Sort key for a version token: v01 < v22 < 'final'."""
    match = _VERSION_RE.search(stem)
    if match:
        return (0, int(match.group(1)))
    if "final" in stem.lower():
        return (1, 0)
    return (0, -1)  # unversioned — treat as earliest


def discover_families(docs_dir: str = DOCS_DIR):
    """Group the PDF corpus into {family: [stems oldest -> newest]}."""
    families = {}
    for name in sorted(os.listdir(docs_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        stem = name[:-4]
        family = re.split(r"[_(]v\d+|_final", stem, flags=re.IGNORECASE)[0].strip("_( ")
        families.setdefault(family, []).append(stem)
    return {
        fam: sorted(stems, key=_version_key)
        for fam, stems in families.items() if len(stems) >= 2
    }


# ---------------------------------------------------------------------------
# Scorer adapters
# ---------------------------------------------------------------------------

def _reset_scoring_state():
    from qmix_report_writer.utils.globals import ReportState, Score, LengthGoal
    ReportState.instance().reset()
    try:
        Score.instance().reset()
        LengthGoal.instance().reset()
    except Exception:
        pass


async def legacy_scorer_adapter(task, chunks) -> dict:
    """Incremental scoring through the CURRENT judges (experiments.eval).

    Frozen protocol from tests/test_scorer.py: append each chunk with a
    placeholder summary, score after every append, final composite = last
    report_score value. `task` is forwarded to the judges when given; the
    benchmark passes None for the corpus PDFs (their commissioned subject is
    unknown — a family stem is not a subject).
    """
    # Local import: this module must stay importable after Stage 2.6 deletes
    # experiments/eval.py (the adapter itself then becomes unusable, which is
    # fine — the v2 adapter takes over).
    import inspect

    from experiments.eval import report_score

    _reset_scoring_state()
    from qmix_report_writer.utils.globals import ReportState, Score

    kwargs = {}
    if task is not None and "task" in inspect.signature(report_score).parameters:
        kwargs["task"] = task

    chunk_scores, composite = [], 0.0
    for i, chunk in enumerate(chunks):
        ReportState.instance().append(chunk, f"Summary placeholder, chunk {i + 1}")
        composite = await report_score(**kwargs)
        chunk_scores.append(Score.instance().micro_scores[-1])
    _reset_scoring_state()
    return {"final_score": float(composite), "chunk_scores": chunk_scores}


def get_adapter(name: str):
    if name == "legacy":
        return legacy_scorer_adapter
    if name == "v2":
        # Stage 2.8: wire the ReportEvaluator-based adapter here.
        from qmix_report_writer.evaluation import benchmark_adapter  # noqa: F401
        return benchmark_adapter
    raise ValueError(f"unknown adapter '{name}'")


# ---------------------------------------------------------------------------
# Benchmark modes (live: need the Ollama judges)
# ---------------------------------------------------------------------------

async def run_ranking(adapter, docs_dir: str = DOCS_DIR) -> dict:
    families = discover_families(docs_dir)
    results = {"mode": "rank", "families": {}, "pairs": [], "chunk_scores": {}}
    for family, stems in families.items():
        scores = {}
        for stem in stems:
            chunks = chunk_text(extract_text(os.path.join(docs_dir, stem + ".pdf")))
            outcome = await adapter(None, chunks)  # corpus PDFs: subject unknown
            scores[stem] = outcome["final_score"]
            # Per-chunk detail: zero-scored chunks are the parse-failure
            # signature (defect A0.3) — keep them inspectable.
            results["chunk_scores"][stem] = outcome.get("chunk_scores", [])
            print(f"  {stem}: {outcome['final_score']:.4f} ({len(chunks)} chunks)")
        results["families"][family] = scores
        for i in range(len(stems)):
            for j in range(i + 1, len(stems)):
                results["pairs"].append({
                    "older": stems[i], "newer": stems[j],
                    "older_score": scores[stems[i]], "newer_score": scores[stems[j]],
                })
    pair_tuples = [(p["older_score"], p["newer_score"]) for p in results["pairs"]]
    results["ranking_accuracy"] = ranking_accuracy(pair_tuples)
    return results


async def run_repeat(adapter, doc_stem: str, k: int, docs_dir: str = DOCS_DIR) -> dict:
    chunks = chunk_text(extract_text(os.path.join(docs_dir, doc_stem + ".pdf")))
    scores = []
    for i in range(k):
        outcome = await adapter(None, chunks)
        scores.append(outcome["final_score"])
        print(f"  run {i + 1}/{k}: {outcome['final_score']:.4f}")
    return {"mode": "repeat", "doc": doc_stem, "scores": scores,
            "sigma": repeat_variance(scores)}


async def run_corruption(adapter, doc_stem: str, docs_dir: str = DOCS_DIR) -> dict:
    text = extract_text(os.path.join(docs_dir, doc_stem + ".pdf"))
    clean = await adapter(None, chunk_text(text))
    results = {"mode": "corrupt", "doc": doc_stem,
               "clean_score": clean["final_score"], "probes": {}}
    print(f"  clean: {clean['final_score']:.4f}")
    for name, corrupt in CORRUPTIONS.items():
        outcome = await adapter(None, chunk_text(corrupt(text)))
        drop = clean["final_score"] - outcome["final_score"]
        results["probes"][name] = {"score": outcome["final_score"], "drop": drop}
        print(f"  {name}: {outcome['final_score']:.4f} (drop {drop:+.4f})")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results(results: dict, adapter_name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"benchmark_{results['mode']}_{adapter_name}_{stamp}")
    results["adapter"] = adapter_name
    results["timestamp"] = stamp
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [f"# Scorer benchmark — {results['mode']} ({adapter_name}, {stamp})", ""]
    if results["mode"] == "rank":
        lines.append(f"**Ranking accuracy: {results['ranking_accuracy']:.3f}** "
                     f"({len(results['pairs'])} ordered pairs)\n")
        for family, scores in results["families"].items():
            lines.append(f"## {family}")
            lines += [f"- {stem}: {score:.4f}" for stem, score in scores.items()]
            lines.append("")
    elif results["mode"] == "repeat":
        lines.append(f"Doc: {results['doc']} — σ = **{results['sigma']:.4f}** "
                     f"over {len(results['scores'])} runs: "
                     + ", ".join(f"{s:.4f}" for s in results["scores"]))
    elif results["mode"] == "corrupt":
        lines.append(f"Doc: {results['doc']} — clean score {results['clean_score']:.4f}\n")
        lines += [f"- {name}: {p['score']:.4f} (drop {p['drop']:+.4f})"
                  for name, p in results["probes"].items()]
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {base}.json / .md")
    return base


def main():
    parser = argparse.ArgumentParser(description="Benchmark the report scorer.")
    parser.add_argument("mode", choices=["rank", "repeat", "corrupt"])
    parser.add_argument("--adapter", default="legacy", choices=["legacy", "v2"])
    parser.add_argument("--doc", help="Document stem (repeat/corrupt modes).")
    parser.add_argument("-k", type=int, default=5, help="Repeat count.")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)
    if args.mode == "rank":
        results = asyncio.run(run_ranking(adapter))
    elif args.mode == "repeat":
        if not args.doc:
            parser.error("--doc is required for repeat mode")
        results = asyncio.run(run_repeat(adapter, args.doc, args.k))
    else:
        if not args.doc:
            parser.error("--doc is required for corrupt mode")
        results = asyncio.run(run_corruption(adapter, args.doc))
    save_results(results, args.adapter)


if __name__ == "__main__":
    main()
