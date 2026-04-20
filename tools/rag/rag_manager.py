import chromadb
from chromadb.utils import embedding_functions

from utils.config import get_llm_config


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

    def query_docs(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        extracted_data = []
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        for doc, meta in zip(documents, metadatas):
            extracted_data.append({
                "content": doc,
                "source": meta.get("source_name", "Unknown Source"),
                "page": meta.get("page_number", "N/A")
            })
            
        return extracted_data
    
    def add_documents(self, documents, metadatas, ids):
        """
        metadatas should be a list of dicts, e.g., 
        [{'source_name': 'Annual_Report.pdf', 'page_number': 12}, ...]
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )