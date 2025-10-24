from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
import os
from typing import List, Dict


class ChromaConversationMemory:
    def __init__(self, persist_dir: str = "./data/chroma_memory"):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = Chroma(
            collection_name="conversation_memory",
            embedding_function=self.embedding,
            persist_directory=persist_dir
        )

    def save_message(self, session_id: str, role: str, content: str):
        """Save a message (user or assistant) into Chroma memory."""
        doc = Document(
            page_content=content,
            metadata={"session_id": session_id, "role": role}
        )
        self.client.add_texts(
            texts=[doc.page_content],
            metadatas=[doc.metadata],
            ids=[f"{session_id}_{role}_{hash(content)}"]
        )
        self.client.persist()

    def load_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve the full conversation history for a given session."""
        results = self.client.get(where={"session_id": session_id})
        if not results or not results.get("documents"):
            return []
        return [
            {"role": meta["role"], "content": doc}
            for doc, meta in zip(results["documents"], results["metadatas"])
            if meta.get("session_id") == session_id
        ]

    def load_memory(self, session_id: str) -> str:
        """Compatibility wrapper to return memory as a formatted text string."""
        messages = self.load_session_history(session_id)
        if not messages:
            return ""
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

    def clear_session(self, session_id: str):
        """Delete all stored messages for a given session."""
        self.client.delete(where={"session_id": session_id})
        self.client.persist()
