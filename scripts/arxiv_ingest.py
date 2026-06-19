#!/usr/bin/env python3
"""
arxiv_ingest.py — Automated arXiv paper ingestion into the RAG database.

Uses Semantic Scholar to search for papers (no rate-limit issues) and
downloads PDFs directly from arxiv.org. Progress is written to
arxiv_ingest_progress.json after every paper so the run can be resumed if
interrupted.

Usage examples:
    # Ingest up to 200 papers on nuclear theory
    python scripts/arxiv_ingest.py --query "nuclear theory" --max-papers 200

    # Sort by citation count instead of relevance
    python scripts/arxiv_ingest.py --query "nuclear theory" --sort citations --max-papers 100

    # Keep PDFs after embedding
    python scripts/arxiv_ingest.py --query "nuclear theory" --keep-pdfs

    # Check what has been ingested so far
    python scripts/arxiv_ingest.py --status

    # Retry only papers that failed last time
    python scripts/arxiv_ingest.py --query "nuclear theory" --retry-failed
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Project root resolution ──────────────────────────────────────────────────
# RAGManager uses a relative path ("./chroma_data"), so we must be at the
# project root when it is imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from qmix_report_writer.tools.rag import RAGManager

# ── Constants ────────────────────────────────────────────────────────────────
PROGRESS_FILE = PROJECT_ROOT / "scripts" / "arxiv_ingest_progress.json"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "arxiv_downloads"

_INSPIRE_URL = "https://inspirehep.net/api/literature"
_INSPIRE_PAGE_SIZE = 100
_INTER_PAGE_DELAY = 2  # seconds between paginated requests (be polite)

_USER_AGENT = "Q-Mix-report-writer/1.0 (academic research; https://github.com/leobeaumont)"

# SSL context backed by certifi's CA bundle (avoids Windows system-store issues).
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

_INSPIRE_SORT_MAP = {
    "relevance": None,       # INSPIRE default
    "citations": "mostcited",
    "date": "mostrecent",
}


# ── Paper data class ─────────────────────────────────────────────────────────

@dataclass
class _Paper:
    entry_id: str           # "http://arxiv.org/abs/2301.12345"
    title: str
    pdf_url: str            # "https://arxiv.org/pdf/2301.12345"
    categories: List[str] = field(default_factory=list)
    published: Optional[datetime] = None


# ── INSPIRE HEP search ───────────────────────────────────────────────────────

def _inspire_fetch_page(query: str, page: int, size: int, sort: Optional[str]) -> dict:
    """Fetch one page of results from the INSPIRE HEP literature API."""
    params: dict = {"q": query, "size": size, "page": page,
                    "fields": "arxiv_eprints,titles,preprint_date"}
    if sort:
        params["sort"] = sort
    req = urllib.request.Request(
        f"{_INSPIRE_URL}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return json.loads(resp.read())


def _inspire_fetch_with_backoff(query: str, page: int, size: int, sort: Optional[str]) -> dict:
    for attempt in range(6):
        try:
            return _inspire_fetch_page(query, page, size, sort)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 5:
                wait = 15 * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)
            else:
                raise
        except (TimeoutError, OSError) as exc:
            if attempt < 5:
                wait = 10 * (attempt + 1)
                print(f"  Network error ({exc}). Waiting {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)
            else:
                raise


def _search(query: str, max_results: int, sort_by: str) -> List[_Paper]:
    """Search INSPIRE HEP and return papers that have a downloadable arXiv PDF.

    Non-arXiv papers are skipped because their INSPIRE records only link to
    paywalled journal pages — there is no freely downloadable PDF for them.
    We always request full pages of _INSPIRE_PAGE_SIZE so that papers without
    arXiv IDs don't prevent reaching max_results.

    When sorting by citations, a date 2000--2026 guard is added automatically
    to avoid pre-arXiv classics (which have no PDF) dominating the first pages.
    """
    inspire_sort = _INSPIRE_SORT_MAP.get(sort_by)
    effective_query = query
    if sort_by == "citations" and "date" not in query.lower():
        effective_query = f"({query}) AND date 2000--2026"

    results: List[_Paper] = []
    seen_ids: set = set()
    page = 1
    total_fetched = 0

    while len(results) < max_results:
        data = _inspire_fetch_with_backoff(effective_query, page, _INSPIRE_PAGE_SIZE, inspire_sort)
        hits = data.get("hits", {})
        items = hits.get("hits", [])
        total_available = hits.get("total", 0)

        for item in items:
            eprints = item.get("metadata", {}).get("arxiv_eprints") or []
            if not eprints:
                continue
            arxiv_id = eprints[0].get("value", "")
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            titles = item.get("metadata", {}).get("titles") or []
            title = (titles[0].get("title", "") if titles else "").replace("\n", " ").strip()
            pub_str = item.get("metadata", {}).get("preprint_date") or ""
            try:
                published = datetime.fromisoformat(pub_str) if pub_str else None
            except ValueError:
                published = None
            categories = [ep.get("categories", [None])[0] for ep in eprints
                          if ep.get("categories")]

            results.append(_Paper(
                entry_id=f"http://arxiv.org/abs/{arxiv_id}",
                title=title,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                categories=[c for c in categories if c],
                published=published,
            ))
            if len(results) >= max_results:
                break

        total_fetched += len(items)
        page += 1
        if not items or total_fetched >= total_available:
            break
        if len(results) < max_results:
            time.sleep(_INTER_PAGE_DELAY)

    return results


# ── PDF download ─────────────────────────────────────────────────────────────

_PDF_DOWNLOAD_DELAY = 3  # seconds between PDF downloads — respect arXiv's crawl policy

def _download_pdf(pdf_url: str, dest: Path) -> None:
    req = urllib.request.Request(pdf_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()

    # arXiv sometimes returns an HTML error page with a 200 status (or serves
    # PostScript as application/x-eprint when there is no PDF source).
    if b"%PDF-" not in data[:10]:
        ct_hint = f" (Content-Type: {content_type})" if content_type else ""
        raise ValueError(f"Response is not a PDF{ct_hint} — paper may only have PostScript/DVI source")

    dest.write_bytes(data)


# ── Progress file helpers ────────────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "papers": {}}


def _save_progress(state: dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _arxiv_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2301.12345' → '2301.12345'"""
    return entry_id.rstrip("/").split("/")[-1]


def _paper_record(paper: _Paper, status: str) -> dict:
    return {
        "status": status,
        "title": paper.title,
        "pdf_url": paper.pdf_url,
        "categories": paper.categories,
        "published": paper.published.isoformat() if paper.published else None,
        "embedded_at": datetime.now().isoformat() if status == "embedded" else None,
        "error": None,
    }


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    state = _load_progress()
    papers = state.get("papers", {})

    by_status: dict = {}
    for p in papers.values():
        s = p["status"]
        by_status[s] = by_status.get(s, 0) + 1

    print("=== Ingestion progress ===")
    for status in ("embedded", "pending", "failed"):
        print(f"  {status:<10} {by_status.get(status, 0)}")
    print(f"  {'total':<10} {len(papers)}")

    failed = [(pid, p) for pid, p in papers.items() if p["status"] == "failed"]
    if failed:
        print("\nFailed papers:")
        for pid, p in failed:
            title = p["title"].replace("\n", " ")[:60]
            print(f"  [{pid}] {title}")
            print(f"         {p.get('error', 'unknown error')}")

    print("\n=== RAG database ===")
    rag = RAGManager()
    docs = rag.list_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)
    print(f"  Documents: {len(docs)}")
    print(f"  Chunks:    {total_chunks}")


def cmd_ingest(args) -> None:
    query: str = args.query
    max_papers: int = args.max_papers
    keep_pdfs: bool = args.keep_pdfs
    retry_failed: bool = args.retry_failed
    sort_by: str = args.sort
    download_dir = Path(args.download_dir) if args.download_dir else DEFAULT_DOWNLOAD_DIR

    download_dir.mkdir(parents=True, exist_ok=True)
    state = _load_progress()
    papers: dict = state.setdefault("papers", {})

    print("Checking existing database entries...")
    rag = RAGManager()
    db_sources = {doc["source_name"] for doc in rag.list_documents()}

    print(f"Searching INSPIRE HEP for '{query}' (max={max_papers}, sort={sort_by})...")
    try:
        results = _search(query, max_papers, sort_by)
    except Exception as exc:
        print(f"ERROR: search failed — {exc}")
        sys.exit(1)
    print(f"Found {len(results)} papers with arXiv IDs.\n")

    n_embedded = n_skipped = n_failed = 0

    for i, paper in enumerate(results, 1):
        pid = _arxiv_id(paper.entry_id)
        filename = f"{pid}.pdf"
        title_short = paper.title[:65]
        tag = f"[{i:>3}/{len(results)}]"

        existing = papers.get(pid, {})

        if existing.get("status") == "embedded":
            print(f"{tag} SKIP  {title_short}")
            n_skipped += 1
            continue

        if existing.get("status") == "failed" and not retry_failed:
            print(f"{tag} SKIP  (failed — use --retry-failed) {title_short}")
            n_skipped += 1
            continue

        if filename in db_sources:
            print(f"{tag} SYNC  {title_short}")
            papers[pid] = _paper_record(paper, status="embedded")
            _save_progress(state)
            n_skipped += 1
            continue

        print(f"{tag} EMBED {title_short}")

        # Mark pending before download so an interrupted run is visible.
        papers[pid] = _paper_record(paper, status="pending")
        _save_progress(state)

        pdf_path = download_dir / filename
        try:
            if not pdf_path.exists():
                _download_pdf(paper.pdf_url, pdf_path)
                time.sleep(_PDF_DOWNLOAD_DELAY)

            result = rag.add_document_from_path(str(pdf_path), chunk_size=512, overlap=50)
            print(f"       → {result['num_chunks']} chunks")

            papers[pid]["status"] = "embedded"
            papers[pid]["embedded_at"] = datetime.now().isoformat()
            papers[pid]["error"] = None
            _save_progress(state)

            if not keep_pdfs:
                pdf_path.unlink(missing_ok=True)

            n_embedded += 1

        except Exception as exc:
            error_msg = str(exc)
            print(f"       → ERROR: {error_msg}")
            papers[pid]["status"] = "failed"
            papers[pid]["error"] = error_msg
            _save_progress(state)
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            n_failed += 1

    print(f"\n=== Done ===")
    print(f"  Embedded this run: {n_embedded}")
    print(f"  Skipped:           {n_skipped}")
    print(f"  Failed:            {n_failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into the project RAG database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", metavar="QUERY",
                        help='INSPIRE HEP search query. Examples: "nuclear theory", '
                             '"arxiv_categories nucl-th", '
                             '"arxiv_categories nucl-th AND t shell model"')
    parser.add_argument("--max-papers", type=int, default=100, metavar="N",
                        help="Maximum papers to fetch (default: 100)")
    parser.add_argument("--sort", choices=["relevance", "citations", "date"], default="relevance",
                        help="Result ordering (default: relevance)")
    parser.add_argument("--keep-pdfs", action="store_true",
                        help="Keep downloaded PDFs after embedding (default: delete them)")
    parser.add_argument("--download-dir", metavar="PATH",
                        help=f"Directory for temporary PDFs (default: {DEFAULT_DOWNLOAD_DIR})")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt papers that failed in a previous run")
    parser.add_argument("--status", action="store_true",
                        help="Show ingestion progress and database state, then exit")

    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    if not args.query:
        parser.error("--query is required unless --status is given")

    cmd_ingest(args)


if __name__ == "__main__":
    main()
