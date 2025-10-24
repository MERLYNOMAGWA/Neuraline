import logging
from chromadb import PersistentClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class ChromaDBClient:
    """Handles connection and operations with Chroma vector database."""
    def __init__(self):
        self.persist_directory = "./data/chroma"
        self.client = PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name="neuraline_knowledge")

    def add_document(self, doc_id: str, text: str, embedding: list):
        try:
            self.collection.add(documents=[text], ids=[doc_id], embeddings=[embedding])
            logger.info(f"✅ Document {doc_id} added to ChromaDB.")
        except Exception as e:
            logger.error(f"❌ Failed to add document {doc_id}: {e}")

    def query(self, query_embedding: list, top_k: int = 3):
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return results
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return None