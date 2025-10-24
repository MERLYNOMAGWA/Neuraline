import logging
import asyncio
from app.services.ai_clients import GeminiClient, GroqClient
from app.services.retriever import ContextRetriever
from app.core.logging_config import log_event

logger = logging.getLogger(__name__)

class ModelRouter:
    """
    Neuraline's Intelligent Model Router
    Dynamically routes user prompts between Gemini and Groq LLMs,
    integrating retrieval-augmented context and fallback recovery.
    """

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()
        self.retriever = ContextRetriever()

    async def run(self, query: str, task_type: str = None, **kwargs):
        """
        Core orchestration method.
        - Classifies task type if not given.
        - Retrieves RAG context when appropriate.
        - Routes intelligently between Gemini and Groq.
        - Includes graceful fallback and robust error handling.
        """
        prompt = query 

        if not task_type:
            task_type = self._classify_task(prompt)
        log_event("MODEL_CALL", f"🧠 Task classified as: {task_type}")

        context = ""
        if task_type in ["rag_query", "emotional_reflection", "cognitive_reasoning"]:
            try:
                log_event("RAG_RETRIEVE", f"🔍 Retrieving context for: {task_type}")
                loop = asyncio.get_event_loop()
                context = await loop.run_in_executor(None, self.retriever.retrieve, prompt)
                if context:
                    log_event("RAG_CONTEXT", f"📚 Retrieved context length: {len(context)} chars")
                    prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}"
            except Exception as e:
                log_event("RAG_ERROR", f"⚠️ Context retrieval failed: {e}")
                prompt = query  

        chosen_model = self._select_model(task_type)
        log_event("MODEL_CALL", f"🎯 Routing to {chosen_model.__class__.__name__} for {task_type}")

        try:
            response = await chosen_model.generate(prompt)
            return response
        except Exception as e:
            log_event("MODEL_ERROR", f"⚠️ Primary model failed: {e}")
            fallback = self._get_fallback(chosen_model)
            log_event("MODEL_FALLBACK", f"🔄 Switching to fallback: {fallback.__class__.__name__}")
            try:
                return await fallback.generate(prompt)
            except Exception as e2:
                log_event("MODEL_FAILSAFE", f"❌ Fallback also failed: {e2}")
                return f"(local fallback) Unable to process with model. Prompt was: {prompt}"#

    def _classify_task(self, prompt: str) -> str:
        """
        Lightweight heuristic classifier for Neuraline's task taxonomy.
        Will later evolve into a classification LLM.
        """
        p = prompt.lower()

        if any(word in p for word in ["reflect", "feeling", "emotion", "why do i feel", "journal"]):
            return "emotional_reflection"
        elif any(word in p for word in ["plan", "goal", "steps", "how to achieve", "strategy"]):
            return "cognitive_reasoning"
        elif any(word in p for word in ["summarize", "analyze", "context", "rag", "retrieve", "neuraline", "explain", "describe"]):
            return "rag_query"
        elif any(word in p for word in ["habit", "track", "consistency", "routine"]):
            return "behavioral_coaching"
        elif any(word in p for word in ["purpose", "mission", "values", "north star"]):
            return "purpose_alignment"
        else:
            return "general_chat"

    def _select_model(self, task_type: str):
        """
        Neuraline's model routing map.
        Gemini → reflective, cognitive, and purpose-driven tasks.
        Groq → behavioral, consistency, and general dialogue.
        """
        model_map = {
            "cognitive_reasoning": self.gemini,
            "emotional_reflection": self.gemini,
            "rag_query": self.gemini,
            "behavioral_coaching": self.groq,
            "purpose_alignment": self.gemini,
            "general_chat": self.groq
        }
        return model_map.get(task_type, self.gemini)

    def _get_fallback(self, model):
        """Return fallback model in case of failure."""
        return self.groq if model == self.gemini else self.gemini