import json
import re
import time
import urllib.request
import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from qmix_report_writer.utils.config import get_llm_config

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


_PAGE_MARKER_RE = re.compile(r"\[PAGE (\d+)\]")

# Embedding token budget. The (remote) Ollama server runs nomic-embed-text with
# an embedding n_ubatch of 512; nomic is an encoder, so llama.cpp cannot split a
# pooled embedding batch — an input over that limit crashes the runner
# (Post .../embedding: EOF -> HTTP 500) instead of erroring cleanly.
#
# We can't count tokens exactly the way the server does: it tokenizes with
# nomic's WordPiece tokenizer, which splits dense math/LaTeX text (physics PDFs)
# into noticeably MORE tokens than tiktoken's cl100k. So we DON'T pre-shrink
# every chunk on a guess. Instead the chunk budget reserves prefix room to keep
# inputs comfortably under 512 cl100k, and _embed_batch self-heals: if the
# server still rejects a batch, it re-embeds that batch one chunk at a time and
# shrinks only the specific chunk(s) that keep failing — good chunks keep their
# full text, and safety no longer depends on the exact cl100k-vs-nomic drift.
_EMBED_TOKEN_LIMIT = 512          # server-side nomic n_ubatch; inputs over this crash the runner
_PREFIX_TOKEN_BUDGET = 24         # headroom for "[Source: <name> | Page: N]\n" prefix
_EMBED_SHRINK_FACTOR = 0.8        # each retry shrinks a faulty input to this fraction of its size
_EMBED_MIN_TOKENS = 64            # floor the self-healing fallback shrinks down to before giving up
_EMBED_RUNNER_RESTART_WAIT = 1.5  # seconds to let the remote runner restart after a crash

# Shared cl100k_base encoder, reused for chunking and the embedding truncation guard.
_ENC = tiktoken.get_encoding("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` cl100k tokens (no-op if already short).

    Only ever applied to the text sent to the embedder; stored chunk text is
    never altered.
    """
    ids = _ENC.encode(text)
    if len(ids) <= max_tokens:
        return text
    return _ENC.decode(ids[:max_tokens])

# Bibliographic-metadata extraction (best effort, source-agnostic).
#
# Design rule: prefer no value over a wrong one. Embedded document-info is
# trusted (after junk filtering); first-page heuristics only fill the gaps it
# leaves and bail out to "" whenever they are not confident.
import datetime as _datetime

# Plausible publication-year window. Future years (e.g. an OCR'd "2072") and
# absurdly old ones are rejected so a stray number never becomes the year.
_MIN_YEAR = 1900
_MAX_YEAR = _datetime.date.today().year + 1
# A standalone 4-digit year (not embedded in a longer number).
_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")

# Lines that can never be a title (arXiv stamp, emails, URLs, bare numbers).
_TITLE_REJECT_RE = re.compile(
    r"(?:^arxiv[:\s]|@|https?://|www\.|^\d+$|preprint|submitted to|^doi[:\s])",
    re.IGNORECASE,
)
# Lowercase function words that mark a line as prose (i.e. a title), so it must
# not be mistaken for an author list. "and" is excluded because it also joins
# authors ("Smith and Jones").
_PROSE_WORDS = frozenset(
    "a an the of for with in on to from via using into under over between "
    "through toward towards is are we this that at by as it its their our".split()
)
# Author-line affiliation noise: superscript/footnote marks and digits.
_AFFIL_NOISE_RE = re.compile(r"[\d∗*†‡§¶•◦]+")
# Obvious junk values seen in embedded document-info.
_JUNK_TITLE_RE = re.compile(
    r"(microsoft word|powerpoint|\.docx?|\.tex|\.dvi|\.pdf|\.indd|untitled|"
    r"^report$|^document$|^title$|^paper$)",
    re.IGNORECASE,
)
_JUNK_AUTHOR_RE = re.compile(
    r"(administrator|^user$|^owner$|^admin$|microsoft|adobe|pdflatex|latex|"
    r"acrobat|writer|default|unknown)",
    re.IGNORECASE,
)


def _clean_meta_value(value) -> str:
    """Normalise a raw metadata string (docinfo or heuristic) for storage."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    # Collapse runs of whitespace introduced by line joins.
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def _valid_year(value: str) -> str:
    """Return the year string if it falls inside the plausible window, else ""."""
    if value and value.isdigit() and _MIN_YEAR <= int(value) <= _MAX_YEAR:
        return value
    return ""


def _year_from_pdf_date(raw) -> str:
    """Extract a plausible 4-digit year from a PDF /CreationDate ('D:20210512...')."""
    if not raw:
        return ""
    m = re.search(r"D:(\d{4})", str(raw))
    if m:
        return _valid_year(m.group(1))
    m = _YEAR_RE.search(str(raw))
    return _valid_year(m.group(0)) if m else ""


def _first_page_lines(text: str) -> List[str]:
    """Return the non-empty lines of the first page (before the [PAGE 2] marker)."""
    cut = text
    second = re.search(r"\[PAGE 2\]", text)
    if second:
        cut = text[: second.start()]
    lines = []
    for raw in cut.splitlines():
        line = raw.strip()
        if not line or _PAGE_MARKER_RE.fullmatch(line):
            continue
        lines.append(line)
    return lines


def _looks_like_title(line: str) -> bool:
    """A line that confidently holds a document title (prose, not an id/code).

    Requires several real words and a prose-like letter mix, which rejects
    report numbers ("YITP-10-28, TKYNT-10-06"), ALL-CAPS banners and
    digit-heavy identifiers.
    """
    if not (15 <= len(line) <= 250) or _TITLE_REJECT_RE.search(line):
        return False
    if len(line.split()) < 3:
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 10:
        return False
    # Titles are mostly lowercase prose; codes/banners are not.
    if sum(c.islower() for c in letters) / len(letters) < 0.45:
        return False
    # Reject digit-dominated lines (identifiers, dates, tabular headers).
    if sum(c.isdigit() for c in line) / len(line) > 0.2:
        return False
    return True


def _clean_author_line(line: str) -> str:
    """Strip affiliation marks/digits and tidy separators from an author line."""
    cleaned = _AFFIL_NOISE_RE.sub(" ", line)
    cleaned = re.sub(r"\s*,\s*(?=and\b|&)", " ", cleaned)  # spurious comma before "and"
    cleaned = re.sub(r"\s+([,;])", r"\1", cleaned)         # space before separator
    cleaned = re.sub(r"([,;])(?=[,;])", "", cleaned)       # collapse repeated separators
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;.")


def _looks_like_authors(line: str) -> bool:
    """A line that confidently holds an author list (person names, not prose)."""
    cleaned = _clean_author_line(line)
    if not (3 <= len(cleaned) <= 200) or _TITLE_REJECT_RE.search(cleaned):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z.'\-]*", cleaned)
    if len(words) < 2:
        return False
    # Any prose function word means this is a title/sentence, not names.
    if any(w.lower() in _PROSE_WORDS for w in words):
        return False
    # The bulk of tokens must look like names (capitalised words or initials).
    namelike = [w for w in words if w[:1].isupper() or (len(w) <= 2 and w[0].isalpha())]
    caps = [w for w in words if w[:1].isupper()]
    return len(caps) >= 2 and len(namelike) / len(words) >= 0.7


def _heuristic_metadata(text: str) -> Tuple[str, str, str]:
    """Best-effort (title, author, year) parsed from first-page text.

    Conservative by design: returns "" for any field it cannot infer confidently.
    Used only to fill gaps left by embedded document-info metadata.
    """
    lines = _first_page_lines(text)
    title = author = ""
    title_idx = -1
    for i, line in enumerate(lines[:15]):
        if _looks_like_title(line):
            title = line
            title_idx = i
            # Stitch an immediately wrapped second title line, but never absorb a
            # line that reads like an author list (a common cause of bad titles).
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if (
                nxt
                and _looks_like_title(nxt)
                and not _looks_like_authors(nxt)
                and len(nxt) < 120
                and not title.endswith((".", ":", "?"))
            ):
                title = f"{title} {nxt}"
            break
    if title_idx >= 0:
        for line in lines[title_idx + 1: title_idx + 4]:
            if line == title or line in title:
                continue
            if _looks_like_authors(line):
                author = _clean_author_line(line)
                break
    # Year: only from the first page, and only inside the plausible window.
    year = ""
    for m in _YEAR_RE.finditer("\n".join(lines)):
        year = _valid_year(m.group(0))
        if year:
            break
    return title, author, year


def _extract_bibliographic_metadata(path: Path, reader_or_doc, text: str) -> Dict[str, str]:
    """Resolve title/author/year: embedded document-info first, heuristics to fill gaps.

    `reader_or_doc` is the already-opened pypdf PdfReader or python-docx Document
    (None for plain text). Embedded values are junk-filtered; first-page
    heuristics only fill what is still missing. Only non-empty fields are
    returned, so callers can add them straight into ChromaDB metadata (which
    rejects None values).
    """
    title = author = year = ""
    suffix = path.suffix.lower()

    if suffix == ".pdf" and reader_or_doc is not None:
        info = reader_or_doc.metadata or {}
        title = _clean_meta_value(info.get("/Title"))
        author = _clean_meta_value(info.get("/Author"))
        year = _year_from_pdf_date(info.get("/CreationDate"))
    elif suffix == ".docx" and reader_or_doc is not None:
        props = reader_or_doc.core_properties
        title = _clean_meta_value(getattr(props, "title", ""))
        author = _clean_meta_value(getattr(props, "author", ""))
        created = getattr(props, "created", None)
        year = _valid_year(str(created.year)) if created else ""

    # Discard obvious docinfo junk so heuristics (or nothing) take over.
    if title and (_JUNK_TITLE_RE.search(title) or not (4 <= len(title) <= 300)):
        title = ""
    if author and (_JUNK_AUTHOR_RE.search(author) or len(author) < 3):
        author = ""

    if not (title and author and year):
        h_title, h_author, h_year = _heuristic_metadata(text)
        title = title or h_title
        author = author or h_author
        year = year or h_year

    out: Dict[str, str] = {}
    if title:
        out["title"] = title
    if author:
        out["author"] = author
    if year:
        out["year"] = year
    return out


def _get_ollama_base_url() -> str:
    """Read the Ollama base URL from default.yaml (same path as ollama_chat.py)."""
    llm_config = get_llm_config()
    return llm_config.get("providers", {}).get("ollama", {}).get("base_url", "http://localhost:11434")


def _load_text_from_file(file_path: str) -> Tuple[str, Dict[str, str]]:
    """Load text from various file formats. Returns (text, metadata)."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = {"source_name": path.name, "file_type": path.suffix.lower()}
    reader_or_doc = None

    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")
        reader = PdfReader(file_path)
        reader_or_doc = reader
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n[PAGE {page_num + 1}]\n"
            text += page.extract_text()

    elif path.suffix.lower() == ".docx":
        if Document is None:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
        doc = Document(file_path)
        reader_or_doc = doc
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    elif path.suffix.lower() in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .pdf, .docx, .txt, .md")

    # Resolve bibliographic metadata (title/author/year) for the bibliography.
    metadata.update(_extract_bibliographic_metadata(path, reader_or_doc, text))
    return text, metadata


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk text by token count with overlap.
    chunk_size: target tokens per chunk
    overlap: tokens to overlap between chunks
    """
    encoding = _ENC
    tokens = encoding.encode(text)

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        start_idx = end_idx - overlap if end_idx < len(tokens) else len(tokens)

    return chunks if chunks else [""]


def _generate_chunk_id(source_name: str, chunk_idx: int) -> str:
    """Generate stable, unique IDs for chunks."""
    return f"{source_name.replace('.', '_')}_chunk_{chunk_idx}"


class RAGManager:
    def __init__(self, collection_name="document_database", rerank_mode: str = "nomic", bm25_floor: int = 1, top_k: int = 5):
        """
        rerank_mode:
          "nomic"          — re-rank RRF candidates by nomic-embed-text cosine similarity (default)
          "cross_encoder"  — re-rank with BAAI/bge-reranker-base (slowest, most accurate)

        bm25_floor:
          Minimum BM25-only chunks guaranteed in the final result.
          Only injects if a BM25-only chunk ranked in the top 10 of the full RRF pool
          (i.e. was genuinely competitive). Falls back to all-vector if none qualify.
          Set to 0 to disable. Default: 1.

        top_k:
          Number of chunks returned per query. Default: 5.
        """
        from qmix_report_writer.utils.config import get_chroma_path
        self.client = chromadb.PersistentClient(path=get_chroma_path())
        """self.client = chromadb.HttpClient(
            host="[IP]",
            port=[PORT]
        )"""

        self.emb_fn = embedding_functions.OllamaEmbeddingFunction(
            url=_get_ollama_base_url(),
            model_name="nomic-embed-text"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )
        self.rerank_mode = rerank_mode
        self.bm25_floor = bm25_floor
        self.top_k = top_k
        self._reranker_model_name = "BAAI/bge-reranker-base"
        self._reranker = None
        self._reranker_tokenizer = None

        self._bm25: Optional[object] = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []
        self._bm25_metas: List[Dict] = []
        self._build_bm25_index()

    def _post_embed(self, base_url: str, inputs: List[str]) -> List[List[float]]:
        """Single Ollama /api/embed call. Raises on any transport/HTTP error."""
        payload = json.dumps({"model": "nomic-embed-text", "input": inputs}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["embeddings"]

    def _embed_batch(self, base_url: str, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one Ollama call, isolating any faulty input.

        Fast path: a single batched request with every chunk at full size. If
        the remote runner rejects the batch (a nomic input over its 512-token
        n_ubatch crashes the runner -> EOF/HTTP 500), we do NOT truncate the
        whole batch. We re-embed it one chunk at a time so the good chunks keep
        their full text, and only the specific chunk(s) that keep failing are
        progressively shrunk (see _embed_one_resilient).
        """
        try:
            return self._post_embed(base_url, list(texts))
        except Exception:
            # The batch crashed the runner; give it a moment to restart, then
            # re-embed chunk by chunk so we can isolate and shrink only offenders.
            time.sleep(_EMBED_RUNNER_RESTART_WAIT)
            return [self._embed_one_resilient(base_url, t) for t in texts]

    def _embed_one_resilient(self, base_url: str, text: str) -> List[float]:
        """Embed a single chunk, shrinking it only if the server keeps rejecting it.

        Tries the chunk at full size first (most chunks in a failed batch are
        innocent — only their batch-mate was oversized). Only the chunk that
        actually crashes the runner is truncated, then retried at progressively
        smaller sizes until it embeds or hits _EMBED_MIN_TOKENS.
        """
        try:
            return self._post_embed(base_url, [text])[0]
        except Exception:
            pass  # this chunk is the offender — shrink just this one

        budget = int(len(_ENC.encode(text)) * _EMBED_SHRINK_FACTOR)
        while budget >= _EMBED_MIN_TOKENS:
            time.sleep(_EMBED_RUNNER_RESTART_WAIT)
            try:
                return self._post_embed(base_url, [_truncate_to_tokens(text, budget)])[0]
            except Exception:
                budget = int(budget * _EMBED_SHRINK_FACTOR)
        raise RuntimeError(
            f"embedding failed even after shrinking to {_EMBED_MIN_TOKENS} tokens"
        )

    def _embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Call Ollama directly for embeddings, bypassing ChromaDB's internal routing.

        Embeds in batches (one API call per batch) instead of one request per
        chunk, which is much faster for ingestion. _embed_batch isolates and
        shrinks any chunk the server rejects, so an oversized chunk fails neither
        its batch-mates nor the document. batch_size=16 matches the request width
        already proven safe by the reranking path.
        """
        base_url = _get_ollama_base_url().rstrip("/")
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self._embed_batch(base_url, texts[start:start + batch_size]))
        return embeddings

    def _build_bm25_index(self) -> None:
        """Build an in-memory BM25 index from the current ChromaDB collection."""
        if not _BM25_AVAILABLE:
            return
        all_data = self.collection.get()
        texts = all_data.get("documents") or []
        self._bm25_ids = all_data.get("ids") or []
        self._bm25_texts = texts
        self._bm25_metas = all_data.get("metadatas") or []
        self._bm25 = BM25Okapi([t.lower().split() for t in texts]) if texts else None

    def _get_candidates_bm25(self, query: str, n_candidates: int) -> List[Dict]:
        """Return top-n BM25 candidates for a query, excluding zero-score results."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_candidates]
        return [
            {
                "content": self._bm25_texts[i],
                "source": self._bm25_metas[i].get("source_name", "Unknown Source"),
                "page": self._bm25_metas[i].get("page_number", "N/A"),
                "title": self._bm25_metas[i].get("title", ""),
                "author": self._bm25_metas[i].get("author", ""),
                "year": self._bm25_metas[i].get("year", ""),
                "distance": None,
                "bm25_score": round(float(scores[i]), 4),
                "id": self._bm25_ids[i],
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def _rrf_merge(self, *candidate_lists: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion across any number of ranked candidate lists.

        score(d) = Σ  1 / (k + rank_i(d) + 1)
        k=60 is the standard constant from the original RRF paper.

        Attaches to each returned chunk:
          rrf_score  — the fused score (higher is better)
          in_vector  — True if this chunk appeared in candidate_lists[0] (vector pool)
          in_bm25    — True if this chunk appeared in candidate_lists[1] (BM25 pool)
        """
        scores: Dict[str, float] = {}
        by_id: Dict[str, Dict] = {}
        pool_membership: Dict[str, set] = {}
        for pool_idx, candidates in enumerate(candidate_lists):
            for rank, c in enumerate(candidates):
                cid = c["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                if cid not in by_id:
                    by_id[cid] = dict(c)
                else:
                    # Merge pool-specific score fields that may only exist in one pool
                    # (e.g. bm25_score lives on BM25 candidates, distance on vector ones)
                    for score_key in ("bm25_score", "distance"):
                        if c.get(score_key) is not None and by_id[cid].get(score_key) is None:
                            by_id[cid][score_key] = c[score_key]
                pool_membership.setdefault(cid, set()).add(pool_idx)
        result = []
        for cid in sorted(scores, key=scores.__getitem__, reverse=True):
            doc = dict(by_id[cid])
            pools = pool_membership.get(cid, set())
            doc["rrf_score"] = round(scores[cid], 6)
            doc["in_vector"] = 0 in pools
            doc["in_bm25"] = 1 in pools
            result.append(doc)
        return result

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Cross-encoder reranking with BAAI/bge-reranker-base."""
        if self._reranker is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
            self._reranker = AutoModelForSequenceClassification.from_pretrained(self._reranker_model_name)
            self._reranker.eval()
        import torch
        pairs = [[query, c["content"]] for c in candidates]
        inputs = self._reranker_tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            scores = self._reranker(**inputs).logits.squeeze(-1).float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = []
        for score, c in ranked[:top_k]:
            doc = dict(c)
            doc["reranker_score"] = round(float(score), 4)
            result.append(doc)
        return result

    def _rerank_nomic(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Re-rank candidates by cosine similarity to the query using nomic-embed-text.

        Embeds the query and all candidate chunks, then sorts by cosine similarity.
        This is lighter than the cross-encoder but can favour semantically similar
        chunks over keyword-match chunks that BM25 surfaced.
        """
        import math
        base_url = _get_ollama_base_url().rstrip("/")
        texts = [query] + [c["content"] for c in candidates]
        embeddings = self._embed_batch(base_url, texts)
        q_emb = embeddings[0]
        q_norm = math.sqrt(sum(x * x for x in q_emb)) or 1.0

        def cosine_sim(emb: List[float]) -> float:
            dot = sum(a * b for a, b in zip(q_emb, emb))
            norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            return dot / (q_norm * norm)

        scored = []
        for c, emb in zip(candidates, embeddings[1:]):
            doc = dict(c)
            doc["nomic_score"] = round(cosine_sim(emb), 4)
            scored.append(doc)

        scored.sort(key=lambda x: x["nomic_score"], reverse=True)
        return scored[:top_k]

    def _get_candidates(self, query_text: str, n_candidates: int, distance_threshold: float) -> List[Dict]:
        """Fetch and filter raw candidates from ChromaDB for a single query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "content": doc,
                "source": meta.get("source_name", "Unknown Source"),
                "page": meta.get("page_number", "N/A"),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "year": meta.get("year", ""),
                "distance": round(dist, 4),
                "id": chunk_id,
            }
            for doc, meta, dist, chunk_id in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
                results.get("ids", [[]])[0],
            )
            if dist <= distance_threshold
        ]

    def _apply_bm25_floor(self, ranked: List[Dict], full_pool: List[Dict], floor: int, top_k: int) -> List[Dict]:
        """Guarantee at least `floor` BM25-only chunks in the final top_k result.

        Only considers BM25-only chunks from the top-10 of the RRF-sorted full_pool —
        if no BM25-only chunk ranked that high, the all-vector result is kept as-is.
        Injects by replacing the lowest-ranked vector-only chunks in `ranked`.
        """
        if floor <= 0:
            return ranked

        result = list(ranked)
        bm25_only_count = sum(1 for c in result if c.get("in_bm25") and not c.get("in_vector"))
        if bm25_only_count >= floor:
            return result

        needed = floor - bm25_only_count
        result_ids = {c["id"] for c in result}

        # Only inject from the top-10 of the RRF pool — low-ranked BM25 chunks are not worth forcing in
        eligible = [
            c for c in full_pool[:10]
            if c.get("in_bm25") and not c.get("in_vector") and c["id"] not in result_ids
        ]
        to_inject = eligible[:needed]
        if not to_inject:
            return result

        for chunk in to_inject:
            # Replace the last (lowest-ranked) vector-only chunk
            for j in range(len(result) - 1, -1, -1):
                if result[j].get("in_vector") and not result[j].get("in_bm25"):
                    result[j] = chunk
                    break
            else:
                if len(result) < top_k:
                    result.append(chunk)

        return result

    def query_docs(self, query_text: str, n_candidates: int = 15, top_k: int = None, distance_threshold: float = 0.7) -> List[Dict]:
        """
        Hybrid retrieval: vector search + BM25 → RRF merge → optional rerank.

        Reranking is controlled by self.rerank_mode (set at construction time):
          None             — return RRF top-k directly
          "nomic"          — re-rank by nomic-embed-text cosine similarity
          "cross_encoder"  — re-rank with BAAI/bge-reranker-base

        Every returned chunk carries debug fields:
          distance      — cosine distance from vector search (None if BM25-only)
          bm25_score    — BM25 score (None if vector-only)
          rrf_score     — RRF fusion score
          in_vector     — True if chunk came from the vector pool
          in_bm25       — True if chunk came from the BM25 pool
          nomic_score   — cosine similarity score when rerank_mode="nomic"
          reranker_score — cross-encoder score when rerank_mode="cross_encoder"
        """
        top_k = top_k if top_k is not None else self.top_k
        count = self.collection.count()
        if count == 0:
            return []
        n = min(n_candidates, count)
        vector_cands = self._get_candidates(query_text, n, distance_threshold)
        bm25_cands = self._get_candidates_bm25(query_text, n)
        merged = self._rrf_merge(vector_cands, bm25_cands)
        if not merged:
            return []
        if self.rerank_mode == "cross_encoder":
            ranked = self._rerank(query_text, merged, top_k)
        else:
            ranked = self._rerank_nomic(query_text, merged, top_k)
        return self._apply_bm25_floor(ranked, merged, self.bm25_floor, top_k)

    def query_docs_multi(self, queries: List[str], top_k: int = None, distance_threshold: float = 0.7) -> List[Dict]:
        """
        Multi-query hybrid retrieval: for each query run vector + BM25, RRF-merge per query,
        then deduplicate across queries and optionally rerank the final pool.

        The first query is used as the anchor for reranking.
        See query_docs for debug field descriptions.
        """
        count = self.collection.count()
        if count == 0:
            return []
        n_candidates = min(15, count)

        top_k = top_k if top_k is not None else self.top_k
        seen_ids: set = set()
        merged: List[Dict] = []
        for query in queries:
            vector_cands = self._get_candidates(query, n_candidates, distance_threshold)
            bm25_cands = self._get_candidates_bm25(query, n_candidates)
            for c in self._rrf_merge(vector_cands, bm25_cands):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    merged.append(c)

        if not merged:
            return []
        if self.rerank_mode == "cross_encoder":
            ranked = self._rerank(queries[0], merged, top_k)
        else:
            ranked = self._rerank_nomic(queries[0], merged, top_k)
        return self._apply_bm25_floor(ranked, merged, self.bm25_floor, top_k)
    
    def add_documents(self, documents, metadatas, ids, embeddings: Optional[List] = None):
        """
        metadatas should be a list of dicts, e.g.,
        [{'source_name': 'Annual_Report.pdf', 'page_number': 12}, ...]

        embeddings: optional pre-computed embeddings (e.g. from contextual prefixing).
                    When provided, ChromaDB stores documents as-is but uses these vectors.
        """
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        self._build_bm25_index()

    def add_document_from_path(self, file_path: str, chunk_size: int = 512, overlap: int = 50) -> Dict:
        """
        Load a document, chunk it, generate metadata, and add to database.

        Page numbers are extracted from [PAGE N] markers already embedded in
        PDF text and stored in chunk metadata so citations are accurate.

        Each chunk is embedded using a contextual prefix
        "[Source: name | Page: N]" prepended to the text, which anchors
        vague chunks to their document context at embedding time. The stored
        document text remains the original (no prefix).

        Args:
            file_path: Path to document (.pdf, .docx, .txt, .md)
            chunk_size: Target tokens per chunk (default 512)
            overlap: Overlap tokens between chunks (default 50)

        Returns:
            Dictionary with ingestion stats: num_chunks, source_name, etc.
        """
        text, base_metadata = _load_text_from_file(file_path)
        # The contextual prefix ("[Source: ... | Page: N]\n") is embedded together
        # with each chunk, so it must fit inside the 512-token encoder limit too.
        # Reserve prefix headroom in the chunk budget so prefix + chunk <= 512 and
        # no real content is lost to the safety-net truncation in _embed_batch.
        effective_chunk_size = min(chunk_size, _EMBED_TOKEN_LIMIT - _PREFIX_TOKEN_BUDGET)
        chunks = _chunk_text(text, chunk_size=effective_chunk_size, overlap=overlap)

        source_name = base_metadata["source_name"]
        file_type = base_metadata["file_type"]
        is_pdf = file_type == ".pdf"
        # Bibliographic fields resolved at load time, replicated onto every chunk
        # so they survive retrieval and reach the bibliography builder.
        bib_fields = {k: base_metadata[k] for k in ("title", "author", "year") if k in base_metadata}

        documents = []
        metadatas = []
        ids = []
        texts_to_embed = []

        current_page = 1
        for idx, chunk in enumerate(chunks):
            # 3.1 — page tracking: update running page whenever a marker appears
            markers = _PAGE_MARKER_RE.findall(chunk)
            if markers:
                current_page = int(markers[-1])

            chunk_metadata = {
                "source_name": source_name,
                "file_type": file_type,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **bib_fields,
            }
            if is_pdf:
                chunk_metadata["page_number"] = current_page

            # 3.2 — contextual prefix for embedding only (original text stored)
            prefix = (
                f"[Source: {source_name} | Page: {current_page}]"
                if is_pdf
                else f"[Source: {source_name}]"
            )
            texts_to_embed.append(f"{prefix}\n{chunk}")

            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(_generate_chunk_id(source_name, idx))

        embeddings = self._embed_texts(texts_to_embed)
        self.add_documents(documents, metadatas, ids, embeddings=embeddings)

        return {
            "success": True,
            "source_name": source_name,
            "num_chunks": len(chunks),
            "file_type": file_type,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }

    def list_documents(self) -> List[Dict]:
        """
        List all documents currently in the database.

        Returns:
            List of dicts with source_name, file_type, and chunk_count.
        """
        all_data = self.collection.get()
        metadatas = all_data.get("metadatas", [])

        doc_info = {}
        for meta in metadatas:
            source = meta.get("source_name", "Unknown")
            if source not in doc_info:
                doc_info[source] = {
                    "source_name": source,
                    "file_type": meta.get("file_type", "unknown"),
                    "chunk_count": 0
                }
            doc_info[source]["chunk_count"] += 1

        return list(doc_info.values())

    def delete_document_by_source(self, source_name: str) -> Dict:
        """
        Delete all chunks of a specific document from the database.

        Args:
            source_name: Name of the source file (e.g., 'ACCADA_final.pdf')

        Returns:
            Dictionary with deletion stats.
        """
        all_data = self.collection.get()
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        ids_to_delete = [
            doc_id for doc_id, meta in zip(ids, metadatas)
            if meta.get("source_name") == source_name
        ]

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            self._build_bm25_index()
            return {
                "success": True,
                "source_name": source_name,
                "chunks_deleted": len(ids_to_delete)
            }
        else:
            return {
                "success": False,
                "source_name": source_name,
                "chunks_deleted": 0,
                "message": f"No document found with source_name: {source_name}"
            }

    def clear_collection(self) -> Dict:
        """
        Clear all documents from the collection (irreversible).

        Returns:
            Dictionary with deletion stats.
        """
        all_data = self.collection.get()
        ids = all_data.get("ids", [])
        chunk_count = len(ids)

        if ids:
            self.collection.delete(ids=ids)
            self._build_bm25_index()
            return {
                "success": True,
                "message": "Collection cleared",
                "chunks_deleted": chunk_count
            }
        else:
            return {
                "success": True,
                "message": "Collection was already empty",
                "chunks_deleted": 0
            }


if __name__ == "__main__":
    manager = RAGManager()

    print("=== Clearing database ===")
    result = manager.clear_collection()
    print(f"  {result['message']}: {result['chunks_deleted']} chunks removed")

    documents = [
        "arxiv_downloads/ACCADA v012.pdf",
    ]

    print("\n=== Ingesting documents with chunk_size=256 ===")
    for doc_path in documents:
        try:
            result = manager.add_document_from_path(doc_path, chunk_size=256, overlap=50)
            print(f"[SUCCESS] Ingested {result['source_name']}")
            print(f"  - Chunks created: {result['num_chunks']}")
            print(f"  - File type: {result['file_type']}")
        except Exception as e:
            print(f"[FAILED] {doc_path}: {e}")

    print("\n=== Final database state ===")
    docs = manager.list_documents()
    total_chunks = 0
    for doc in docs:
        print(f"  {doc['source_name']} ({doc['file_type']}): {doc['chunk_count']} chunks")
        total_chunks += doc['chunk_count']
    print(f"\nTotal chunks in database: {total_chunks}")
