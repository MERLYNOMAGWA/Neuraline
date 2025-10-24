import logging
from app.services.ai_clients import EmbeddingClient
from app.services.vector_store import ChromaDBClient

logger = logging.getLogger(__name__)

class ContextRetriever:
    """Retrieves contextually relevant data from Chroma for RAG reasoning."""

    def __init__(self):
        self.db = ChromaDBClient()
        self.embedder = EmbeddingClient()

    def retrieve(self, query: str, top_k: int = 3):
        query_emb = self.embedder.embed(query)
        results = self.db.query(query_emb, top_k)
        if not results or not results.get("documents"):
            return ""
        documents = [doc for sublist in results["documents"] for doc in sublist]
        return "\n".join(documents)