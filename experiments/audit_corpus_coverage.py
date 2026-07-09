"""
Corpus-coverage audit for the training task list -- RAG-only, no agent LLM calls.

For each task subject, run the same hybrid retrieval the Researcher's PLANNING
scan uses (vector + BM25 + nomic rerank) and report how much grounded evidence
exists. Flags subjects too thin to support a grounded report.

Doubles as a post-transfer sanity check on a new machine: the chunk count and
per-subject verdicts should match the source machine.

Usage (from the project root):
    python experiments/audit_corpus_coverage.py
    python experiments/audit_corpus_coverage.py --top-k 12 --strong 0.5
"""

import os
import sys
import argparse
import statistics as stats

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tasks import tasks
from qmix_report_writer.tools.rag import RAGManager


def score_of(d):
    for key in ("nomic_score", "reranker_score", "rrf_score"):
        v = d.get(key)
        if v is not None:
            return float(v)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=12,
                    help="chunks to retrieve per subject (probe breadth, not the pipeline's k)")
    ap.add_argument("--strong", type=float, default=0.5,
                    help="nomic rerank score counted as a solidly-relevant chunk")
    args = ap.parse_args()

    rag = RAGManager(rerank_mode="nomic", bm25_floor=1)
    print(f"ChromaDB chunks in collection: {rag.collection.count()}\n")

    print("=" * 100)
    print(f"{'#':>2}  {'task':<52} {'chunks':>6} {'srcs':>5} {'strong':>6} "
          f"{'topScore':>8} {'medScore':>8}  verdict")
    print("=" * 100)

    thin = []
    for i, task in enumerate(tasks):
        subject = task.strip()
        docs = rag.query_docs_multi([subject], top_k=args.top_k)
        n = len(docs)
        srcs = len({d.get("source") for d in docs})
        scores = sorted((score_of(d) for d in docs), reverse=True)
        strong = sum(1 for s in scores if s >= args.strong)
        top = scores[0] if scores else 0.0
        med = stats.median(scores) if scores else 0.0
        if n == 0:
            verdict = "EMPTY (will abort)"
        elif strong >= 3:
            verdict = "OK"
        elif strong >= 1 or n >= 3:
            verdict = "THIN"
        else:
            verdict = "VERY THIN"
        if verdict != "OK":
            thin.append((i, subject, verdict, strong, n))
        print(f"{i:>2}  {subject[:52]:<52} {n:>6} {srcs:>5} {strong:>6} "
              f"{top:>8.3f} {med:>8.3f}  {verdict}")

    print("=" * 100)
    print(f"{len(tasks) - len(thin)}/{len(tasks)} OK")
    if thin:
        print("\nSubjects to reword or drop for a cleaner training set:")
        for i, subj, verdict, strong, n in thin:
            print(f"  [{i}] {subj}  ->  {verdict} ({strong} strong / {n} chunks)")


if __name__ == "__main__":
    main()
