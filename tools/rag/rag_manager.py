import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from pathlib import Path
from typing import List, Dict, Tuple

from utils.config import get_llm_config

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def _build_ollama_embedding_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/api/embeddings"):
        return base_url
    if base_url.endswith("/api"):
        return f"{base_url}/embeddings"
    return f"{base_url}/api/embeddings"


def _get_ollama_embedding_endpoint() -> str:
    """Read the Ollama base URL from default.yaml at call time."""
    llm_config = get_llm_config()
    base_url = llm_config.get("ollama_base_url", "http://localhost:11434")
    return _build_ollama_embedding_endpoint(base_url)


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
            url=_get_ollama_embedding_endpoint(),
            model_name="nomic-embed-text"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )
        self._reranker = None
        self._reranker_tokenizer = None
        self._reranker_model_name = "BAAI/bge-reranker-base"

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
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
        Two-stage retrieval: broad vector search → similarity filter → cross-encoder rerank.

        n_candidates:       chunks fetched from ChromaDB (wide net)
        top_k:              chunks returned after reranking (tight output)
        distance_threshold: cosine distance ceiling (0=identical, 1=orthogonal)
        """
        count = self.collection.count()
        if count == 0:
            return []
        candidates = self._get_candidates(query_text, min(n_candidates, count), distance_threshold)
        if not candidates:
            return []
        return self._rerank(query_text, candidates, top_k)

    def query_docs_multi(self, queries: List[str], top_k: int = 3, distance_threshold: float = 0.7) -> List[Dict]:
        """
        Multi-query retrieval: gather candidates from each query, deduplicate by chunk ID, rerank.

        queries:            list of semantically distinct search strings
        top_k:              chunks returned after reranking
        distance_threshold: cosine distance ceiling applied per query before merging

        The first query is used as the anchor for cross-encoder reranking.
        """
        count = self.collection.count()
        if count == 0:
            return []
        n_candidates = min(15, count)

        seen_ids: set = set()
        merged: List[Dict] = []
        for query in queries:
            for candidate in self._get_candidates(query, n_candidates, distance_threshold):
                if candidate["id"] not in seen_ids:
                    seen_ids.add(candidate["id"])
                    merged.append(candidate)

        if not merged:
            return []
        return self._rerank(queries[0], merged, top_k)
    
    def add_documents(self, documents, metadatas, ids):
        """
        metadatas should be a list of dicts, e.g.,
        [{'source_name': 'Annual_Report.pdf', 'page_number': 12}, ...]
        """
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def add_document_from_path(self, file_path: str, chunk_size: int = 512, overlap: int = 50) -> Dict:
        """
        Load a document, chunk it, generate metadata, and add to database.

        Args:
            file_path: Path to document (.pdf, .docx, .txt, .md)
            chunk_size: Target tokens per chunk (default 512)
            overlap: Overlap tokens between chunks (default 50)

        Returns:
            Dictionary with ingestion stats: num_chunks, source_name, etc.
        """
        text, base_metadata = _load_text_from_file(file_path)
        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "source_name": base_metadata["source_name"],
                "file_type": base_metadata["file_type"],
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(_generate_chunk_id(base_metadata["source_name"], idx))

        self.add_documents(documents, metadatas, ids)

        return {
            "success": True,
            "source_name": base_metadata["source_name"],
            "num_chunks": len(chunks),
            "file_type": base_metadata["file_type"],
            "chunk_size": chunk_size,
            "overlap": overlap
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
