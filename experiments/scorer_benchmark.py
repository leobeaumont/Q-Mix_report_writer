"""
Scorer benchmark harness (training_eval_upgrade_plan.md Stage 0.2, report A2.2).

Measures the report evaluator — BEFORE and AFTER the Part-A rework — on the
versioned real papers in tests/test_documents/:

  * ranking accuracy — over all ordered same-paper version pairs, the fraction
    where the newer (better) version scores higher;
  * repeat variance  — score the same document k times, per-run σ (the
    reward's noise floor);
  * corruption probes — off-topic section injected / sections shuffled /
    numbers perturbed: the score must drop vs the clean run;
  * grounding probe (plan 2.10) — the corpus PDFs carry no stored sources, so
    the claim check + grounding factor (TD3) are otherwise untestable here:
    each numeric chunk is scored once against ITSELF as the source (expect
    supported claims) and once against a numbers-perturbed copy (expect
    contradicted claims and a score drop).

The scoring path is a pluggable async *adapter* `(task, chunks) -> result
dict`. The default `v2` adapter runs the Stage-2 grounded `ReportEvaluator`
(Stage 2.8 re-run). The legacy adapter was removed with experiments/eval.py
in Stage 2.6 — the Stage 0.3/1.4 baselines it produced are archived in
tests/scoring_results/.

Metric functions are pure and offline-tested (test_training_eval_acceptance
Stage 0.2); actually scoring documents needs the live Ollama judges.

Usage (from the repo root, Ollama serving the judge model):
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py rank [--self-sources] [--derive-subject]
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py repeat --doc "Towards(v22)" -k 5
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py corrupt --doc "Towards(v22)" [--derive-subject]
    .venv\\Scripts\\python.exe experiments\\scorer_benchmark.py ground --doc "Towards(v22)" [--max-chunks 8]
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


def derive_subject(text: str, max_words: int = 20) -> str:
    """A crude per-document subject: the first words of the extracted text
    (PDF extraction starts at the title). Keeps the judges from being
    subject-blind on corpus documents that carry no commissioned task
    (plan 2.10 — task absence is a benchmark artifact, not a scorer input)."""
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    return " ".join(words[:max_words]).strip()


def grounding_probe_summary(rows):
    """Aggregate grounding-probe rows (pure; offline-tested).

    Each row: {"supporting": {...}, "contradicting": {...}} where both passes
    carry score / n_claims / n_supported / n_unsupported / n_contradicted /
    grounding_ratio for the SAME chunk (self-source vs perturbed-source).

      * claims_fired_fraction — chunks where BOTH passes extracted >= 1 claim
        (the plan-2.9 health metric: grounding actually engages);
      * mean_grounding_supporting — mean grounding_ratio on the supporting
        pass (None when no pass produced a ratio);
      * contradiction_detected_fraction — contradicting passes with >= 1
        contradicted verdict;
      * mean_contradicted_claims — mean n_contradicted on the contradicting pass;
      * mean_score_drop — mean(supporting.score - contradicting.score); must
        be positive for TD3 to punish factually-wrong content.
      * numeric_claim_fraction — digit-carrying claims / all claims on the
        supporting pass. Interprets the rest: number-perturbation can only
        create contradictions in NUMERIC claims, so a low fraction means the
        probe result reflects the chunk's prose, not verdicting quality
        (live finding 2026-07-08: qualitative claims are correctly supported
        by a numbers-perturbed source).
    """
    if not rows:
        return {"n_probed": 0, "claims_fired_fraction": 0.0,
                "mean_grounding_supporting": None,
                "contradiction_detected_fraction": 0.0,
                "mean_contradicted_claims": 0.0, "mean_score_drop": 0.0,
                "numeric_claim_fraction": None}
    fired = sum(1 for r in rows if r["supporting"]["n_claims"] > 0
                and r["contradicting"]["n_claims"] > 0)
    ratios = [r["supporting"]["grounding_ratio"] for r in rows
              if r["supporting"]["grounding_ratio"] is not None]
    detected = sum(1 for r in rows if r["contradicting"]["n_contradicted"] > 0)
    total_claims = sum(r["supporting"].get("n_claims", 0) for r in rows)
    numeric_claims = sum(r["supporting"].get("n_numeric_claims", 0) for r in rows)
    return {
        "n_probed": len(rows),
        "claims_fired_fraction": fired / len(rows),
        "mean_grounding_supporting": (sum(ratios) / len(ratios)) if ratios else None,
        "contradiction_detected_fraction": detected / len(rows),
        "mean_contradicted_claims": (
            sum(r["contradicting"]["n_contradicted"] for r in rows) / len(rows)),
        "mean_score_drop": (
            sum(r["supporting"]["score"] - r["contradicting"]["score"]
                for r in rows) / len(rows)),
        "numeric_claim_fraction": (
            numeric_claims / total_claims if total_claims else None),
    }


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


def reorder_sentences(piece: str) -> str:
    """Sentence-reversed copy: same facts, different surface (pure).

    The grounding probe's source must NOT be a verbatim copy of the chunk:
    live-observed (2026-07-08), the judge treats near-identical texts as "the
    same text" and copies the claim as its own evidence without looking the
    value up — training sources are paraphrases, never byte-copies, so the
    probe must not be one either.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", piece or "") if s.strip()]
    if len(sentences) < 2:
        return (piece or "").strip()
    return " ".join(reversed(sentences))


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

async def evaluator_scorer_adapter(task, chunks, evaluator=None,
                                   sources_per_chunk=None) -> dict:
    """Scoring through the v2 grounded evaluator (Stage 2, TD1 shape).

    Each chunk is scored on its own (`score_chunk`; the corpus PDFs carry no
    stored sources, so by default this exercises the audit-only path — pass
    `sources_per_chunk` to engage the claim check, plan 2.10), then ONE
    terminal macro call scores the whole document — the same per-chunk +
    terminal structure the training reward uses. Final composite mirrors the
    legacy weighting for comparability: 0.3 * macro + 0.7 * mean(chunk scores).

    `evaluator` is injectable for tests; None builds the config default.
    """
    from qmix_report_writer.evaluation import ReportEvaluator

    if evaluator is None:
        evaluator = ReportEvaluator()

    chunk_scores, failures = [], 0
    for i, chunk in enumerate(chunks):
        result = await evaluator.score_chunk(
            chunk=chunk,
            sources=(sources_per_chunk[i] if sources_per_chunk else []),
            task=task, context=None,
        )
        if result is None:
            # Mirror the training controller (plan 1.3): a failed judge skips
            # this chunk's measurement instead of killing an hours-long run.
            failures += 1
            chunk_scores.append(None)
            print(f"  [judge failure on chunk {i + 1}/{len(chunks)}]")
        else:
            chunk_scores.append(result.score)

    valid = [s for s in chunk_scores if s is not None]
    if chunks and not valid:
        raise RuntimeError(
            f"all {failures} chunks failed to score — judge/endpoint is down; "
            f"aborting instead of reporting a fake score"
        )

    macro = await evaluator.score_report(
        task=task or "", outline=[], report="\n\n".join(chunks),
    )
    if macro is None:
        failures += 1
        print("  [judge failure on the terminal macro call]")

    micro_mean = sum(valid) / len(valid) if valid else 0.0
    macro_score = macro.score if macro is not None else 0.0
    return {
        "final_score": 0.3 * macro_score + 0.7 * micro_mean,
        "macro_score": macro_score,
        "chunk_scores": chunk_scores,
        "judge_failures": failures,
    }


def get_adapter(name: str):
    if name == "v2":
        return evaluator_scorer_adapter
    if name == "legacy":
        raise ValueError(
            "the legacy scorer was removed in Stage 2.6 (experiments/eval.py "
            "deleted); its baseline results are archived in tests/scoring_results/"
        )
    raise ValueError(f"unknown adapter '{name}'")


# ---------------------------------------------------------------------------
# Benchmark modes (live: need the Ollama judges)
# ---------------------------------------------------------------------------

async def run_ranking(adapter, docs_dir: str = DOCS_DIR,
                      self_sources: bool = False,
                      use_derived_subject: bool = False) -> dict:
    families = discover_families(docs_dir)
    results = {"mode": "rank", "families": {}, "pairs": [], "chunk_scores": {},
               "judge_failures": {}, "self_sources": self_sources,
               "derived_subject": use_derived_subject}
    for family, stems in families.items():
        scores = {}
        for stem in stems:
            text = extract_text(os.path.join(docs_dir, stem + ".pdf"))
            chunks = chunk_text(text)
            task = derive_subject(text) if use_derived_subject else None
            sources = ([[{"source": "reordered_excerpt",
                          "content": reorder_sentences(c)}] for c in chunks]
                       if self_sources else None)
            outcome = await adapter(task, chunks, sources_per_chunk=sources)
            scores[stem] = outcome["final_score"]
            # Per-chunk detail: zero-scored chunks are the parse-failure
            # signature (defect A0.3) — keep them inspectable.
            results["chunk_scores"][stem] = outcome.get("chunk_scores", [])
            results["judge_failures"][stem] = outcome.get("judge_failures", 0)
            fail_note = (f", {outcome['judge_failures']} judge failure(s)"
                         if outcome.get("judge_failures") else "")
            print(f"  {stem}: {outcome['final_score']:.4f} ({len(chunks)} chunks{fail_note})")
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


async def run_corruption(adapter, doc_stem: str, docs_dir: str = DOCS_DIR,
                         use_derived_subject: bool = False) -> dict:
    text = extract_text(os.path.join(docs_dir, doc_stem + ".pdf"))
    # A derived subject un-blinds subject_coverage — in 2.8 the off-topic
    # probe stayed flat precisely because the judges had no subject.
    task = derive_subject(text) if use_derived_subject else None
    clean = await adapter(task, chunk_text(text))
    results = {"mode": "corrupt", "doc": doc_stem,
               "derived_subject": use_derived_subject,
               "clean_score": clean["final_score"], "probes": {}}
    print(f"  clean: {clean['final_score']:.4f}")
    for name, corrupt in CORRUPTIONS.items():
        outcome = await adapter(task, chunk_text(corrupt(text)))
        drop = clean["final_score"] - outcome["final_score"]
        results["probes"][name] = {"score": outcome["final_score"], "drop": drop}
        print(f"  {name}: {outcome['final_score']:.4f} (drop {drop:+.4f})")
    return results


async def run_grounding(doc_stem: str, docs_dir: str = DOCS_DIR,
                        max_chunks: int = 8, text: str = None,
                        evaluator=None, probe_chars: int = 1500) -> dict:
    """Grounding probe (plan 2.10): per numeric chunk, one scoring pass
    against ITSELF as the source (claims should come back supported) and one
    against a numbers-perturbed copy (claims should come back contradicted,
    score should drop). This is the only benchmark mode that exercises the
    claim check — the corpus carries no stored sources.

    Probed pieces are trimmed to `probe_chars` (~ a training section, not a
    4000-char benchmark chunk): the probe measures the evaluator under the
    conditions the TRAINING reward sees.

    `text`/`evaluator` are injectable for offline tests.
    """
    from qmix_report_writer.evaluation import ReportEvaluator

    if evaluator is None:
        evaluator = ReportEvaluator()
    if text is None:
        text = extract_text(os.path.join(docs_dir, doc_stem + ".pdf"))
    subject = derive_subject(text)
    numeric = [(i, c[:probe_chars]) for i, c in enumerate(chunk_text(text))
               if re.search(r"\d", c[:probe_chars])][:max_chunks]

    def _row(cs):
        # `claims` items double as diagnostics: only NUMERIC claims can be
        # contradicted by number-perturbation, so their count says whether a
        # 0-contradicted result indicts the verdicting or just the prose.
        items = getattr(cs, "claims", []) or []
        return {"score": cs.score, "n_claims": cs.n_claims,
                "n_supported": cs.n_supported, "n_unsupported": cs.n_unsupported,
                "n_contradicted": cs.n_contradicted,
                "grounding_ratio": cs.grounding_ratio,
                "n_numeric_claims": sum(
                    1 for it in items if re.search(r"\d", it.get("claim", ""))),
                "claims": items}

    rows, failures = [], 0
    for i, chunk in numeric:
        # Paraphrase-like sources (sentence-reversed): a verbatim self-source
        # defeats evidence lookup — see reorder_sentences.
        support_src = reorder_sentences(chunk)
        sup = await evaluator.score_chunk(
            chunk=chunk,
            sources=[{"source": "reordered_excerpt", "content": support_src}],
            task=subject, context=None)
        con = await evaluator.score_chunk(
            chunk=chunk,
            sources=[{"source": "perturbed_excerpt",
                      "content": perturb_numbers(support_src)}],
            task=subject, context=None)
        if sup is None or con is None:
            failures += 1
            print(f"  [judge failure on chunk {i}]")
            continue
        sup_row, con_row = _row(sup), _row(con)
        rows.append({"chunk_index": i, "supporting": sup_row,
                     "contradicting": con_row})
        print(f"  chunk {i}: support {sup.score:.4f} "
              f"({sup.n_supported}s/{sup.n_unsupported}u/{sup.n_contradicted}c, "
              f"{sup_row['n_numeric_claims']} numeric) | "
              f"perturbed {con.score:.4f} ({con.n_contradicted} contradicted)")
    if numeric and not rows:
        raise RuntimeError(
            "all probed chunks failed to score — judge/endpoint is down; "
            "aborting instead of reporting a fake probe"
        )

    results = {"mode": "ground", "doc": doc_stem, "subject": subject,
               "judge_failures": failures, "rows": rows}
    results.update(grounding_probe_summary(rows))
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
    elif results["mode"] == "ground":
        mg = results["mean_grounding_supporting"]
        lines.append(f"Doc: {results['doc']} — subject: \"{results['subject'][:80]}\"\n")
        lines += [
            f"- probed chunks: {results['n_probed']} "
            f"(judge failures: {results['judge_failures']})",
            f"- claims fired in both passes: {results['claims_fired_fraction']:.2f}",
            f"- mean grounding ratio (supporting pass): "
            + ("n/a" if mg is None else f"{mg:.3f}"),
            f"- contradiction detected: {results['contradiction_detected_fraction']:.2f} "
            f"(mean {results['mean_contradicted_claims']:.2f} contradicted claims)",
            f"- mean score drop on contradiction: {results['mean_score_drop']:+.4f}",
            f"- numeric claims (the only perturbable ones): "
            + ("n/a" if results.get("numeric_claim_fraction") is None
               else f"{results['numeric_claim_fraction']:.2f} of extracted claims"),
        ]
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {base}.json / .md")
    return base


def main():
    parser = argparse.ArgumentParser(description="Benchmark the report scorer.")
    parser.add_argument("mode", choices=["rank", "repeat", "corrupt", "ground"])
    parser.add_argument("--adapter", default="v2", choices=["legacy", "v2"])
    parser.add_argument("--doc", help="Document stem (repeat/corrupt/ground modes).")
    parser.add_argument("-k", type=int, default=5, help="Repeat count.")
    parser.add_argument("--max-chunks", type=int, default=8,
                        help="Ground mode: numeric chunks probed per document.")
    parser.add_argument("--probe-chars", type=int, default=1500,
                        help="Ground mode: probed piece size (~ a training "
                             "section; the evaluator is measured at the scale "
                             "the training reward sees).")
    parser.add_argument("--self-sources", action="store_true",
                        help="Rank mode: give each chunk itself as its source "
                             "so the claim check engages (plan 2.10).")
    parser.add_argument("--derive-subject", action="store_true",
                        help="Rank/corrupt modes: derive a per-PDF subject "
                             "from the document's first words.")
    args = parser.parse_args()

    if args.mode == "ground":
        if not args.doc:
            parser.error("--doc is required for ground mode")
        results = asyncio.run(run_grounding(args.doc, max_chunks=args.max_chunks,
                                            probe_chars=args.probe_chars))
        save_results(results, args.adapter)
        return

    adapter = get_adapter(args.adapter)
    if args.mode == "rank":
        results = asyncio.run(run_ranking(
            adapter, self_sources=args.self_sources,
            use_derived_subject=args.derive_subject))
    elif args.mode == "repeat":
        if not args.doc:
            parser.error("--doc is required for repeat mode")
        results = asyncio.run(run_repeat(adapter, args.doc, args.k))
    else:
        if not args.doc:
            parser.error("--doc is required for corrupt mode")
        results = asyncio.run(run_corruption(
            adapter, args.doc, use_derived_subject=args.derive_subject))
    save_results(results, args.adapter)


if __name__ == "__main__":
    main()
