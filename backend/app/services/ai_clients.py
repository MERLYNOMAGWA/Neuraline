from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    """Wrapper for Google's Gemini model."""
    def __init__(self):
        self.client = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key
        )
    
    async def generate(self, prompt: str) -> str:
        try:
            response = self.client.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            raise

class GroqClient:
    """Fallback LLM using Groq."""
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
    
    async def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise

class EmbeddingClient:
    """HuggingFace embeddings client."""
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def embed(self, text: str):
        try:
            return self.embedder.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise