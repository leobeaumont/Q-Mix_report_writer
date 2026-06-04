import json
import os
import re
import urllib.request
import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils.config import get_llm_config

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
# Observed character ceiling for nomic-embed-text on this Ollama deployment.
# The embedding input is truncated to this length; stored chunk text is never truncated.
_EMBED_MAX_CHARS = 2900


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

    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n[PAGE {page_num + 1}]\n"
            text += page.extract_text()
        return text, metadata

    elif path.suffix.lower() == ".docx":
        if Document is None:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text, metadata

    elif path.suffix.lower() in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, metadata

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .pdf, .docx, .txt, .md")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk text by token count with overlap.
    chunk_size: target tokens per chunk
    overlap: tokens to overlap between chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
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
    def __init__(self, collection_name="document_database"):
        self.client = chromadb.PersistentClient(path="./chroma_data")
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
        self._reranker = None
        self._reranker_tokenizer = None
        self._reranker_model_name = "BAAI/bge-reranker-base"

        self._bm25: Optional[object] = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []
        self._bm25_metas: List[Dict] = []
        self._build_bm25_index()

    def _embed_one(self, base_url: str, text: str) -> List[float]:
        """Embed a single text, truncating to _EMBED_MAX_CHARS if needed."""
        payload = json.dumps(
            {"model": "nomic-embed-text", "input": [text[:_EMBED_MAX_CHARS]]}
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["embeddings"][0]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama directly for embeddings, bypassing ChromaDB's internal routing.

        Embeds one text at a time with exponential back-off retry to handle
        Ollama's internal child-process restart failures gracefully.
        """
        base_url = _get_ollama_base_url().rstrip("/")
        return [self._embed_one(base_url, text) for text in texts]

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
                "distance": None,
                "id": self._bm25_ids[i],
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def _rrf_merge(self, *candidate_lists: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion across any number of ranked candidate lists.

        score(d) = Σ  1 / (k + rank_i(d) + 1)
        k=60 is the standard constant from the original RRF paper.
        """
        scores: Dict[str, float] = {}
        by_id: Dict[str, Dict] = {}
        for candidates in candidate_lists:
            for rank, c in enumerate(candidates):
                cid = c["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                if cid not in by_id:
                    by_id[cid] = c
        return [by_id[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if self._reranker is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # Suppress model-load noise at the file-descriptor level.
            # contextlib.redirect_* only patches sys.stdout/stderr (Python objects);
            # the safetensors Rust extension and huggingface_hub write directly to
            # fd 1 / fd 2, bypassing Python's I/O layer entirely.
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            _devnull = os.open(os.devnull, os.O_WRONLY)
            _saved_out = os.dup(1)
            _saved_err = os.dup(2)
            try:
                os.dup2(_devnull, 1)
                os.dup2(_devnull, 2)
                self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
                self._reranker = AutoModelForSequenceClassification.from_pretrained(self._reranker_model_name)
            finally:
                os.dup2(_saved_out, 1)
                os.dup2(_saved_err, 2)
                os.close(_saved_out)
                os.close(_saved_err)
                os.close(_devnull)
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
        return [c for _, c in ranked[:top_k]]

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

    def query_docs(self, query_text: str, n_candidates: int = 15, top_k: int = 3, distance_threshold: float = 0.7) -> List[Dict]:
        """
        Three-stage hybrid retrieval: vector search + BM25 → RRF merge → cross-encoder rerank.

        n_candidates:       chunks fetched per retriever (wide net)
        top_k:              chunks returned after reranking (tight output)
        distance_threshold: cosine distance ceiling for the vector path
        """
        count = self.collection.count()
        if count == 0:
            return []
        n = min(n_candidates, count)
        vector_cands = self._get_candidates(query_text, n, distance_threshold)
        bm25_cands = self._get_candidates_bm25(query_text, n)
        merged = self._rrf_merge(vector_cands, bm25_cands)
        if not merged:
            return []
        return self._rerank(query_text, merged, top_k)

    def query_docs_multi(self, queries: List[str], top_k: int = 3, distance_threshold: float = 0.7) -> List[Dict]:
        """
        Multi-query hybrid retrieval: for each query run vector + BM25, RRF-merge per query,
        then deduplicate across queries and cross-encoder rerank the final pool.

        The first query is used as the anchor for cross-encoder reranking.
        """
        count = self.collection.count()
        if count == 0:
            return []
        n_candidates = min(15, count)

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
        return self._rerank(queries[0], merged, top_k)
    
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
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        source_name = base_metadata["source_name"]
        file_type = base_metadata["file_type"]
        is_pdf = file_type == ".pdf"

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
