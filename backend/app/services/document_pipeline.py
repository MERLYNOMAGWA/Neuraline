import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.ai_clients import EmbeddingClient
from app.services.vector_store import ChromaDBClient

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Processes and stores documents for RAG context awareness."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.embedder = EmbeddingClient()
        self.db = ChromaDBClient()

    def process_and_store(self, text: str, source: str):
        chunks = self.splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            emb = self.embedder.embed(chunk)
            self.db.add_document(f"{source}_{i}", chunk, emb)
        logger.info(f"✅ Document from {source} processed and stored.")