import logging
from app.services.ai_clients import GeminiClient, GroqClient
from app.core.logging_config import log_event

logger = logging.getLogger(__name__)

class ModelRouter:
    """
    Intelligent model routing for Neuraline 
    Routes prompts to the most suitable model based on intent and context.
    """

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()

    async def run(self, prompt: str) -> str:
        """
        Routes prompt intelligently between Gemini and Groq.
        Includes fallback and lightweight task-type classification.
        """
        task_type = self._classify_task(prompt)
        log_event("MODEL_CALL", f"🧠 Task classified as: {task_type}")

        chosen_model = self._select_model(task_type)
        log_event("MODEL_CALL", f"🎯 Routing to {chosen_model.__class__.__name__} for {task_type}")

        try:
            response = await chosen_model.generate(prompt)
            return response
        except Exception as e:
            log_event("MODEL_ERROR", f"⚠️ Primary model failed: {e}")
            fallback = self._get_fallback(chosen_model)
            log_event("MODEL_FALLBACK", f"🔄 Switching to fallback: {fallback.__class__.__name__}")
            return await fallback.generate(prompt)


    def _classify_task(self, prompt: str) -> str:
        """
        Heuristic-based lightweight task classifier.
        Later this can be replaced with a small classification LLM.
        """
        p = prompt.lower()

        if any(word in p for word in ["reflect", "feeling", "emotion", "why do i feel", "journal"]):
            return "emotional_reflection"
        elif any(word in p for word in ["plan", "goal", "steps", "how to achieve", "strategy"]):
            return "cognitive_reasoning"
        elif any(word in p for word in ["summarize", "analyze", "context", "rag", "retrieve"]):
            return "rag_query"
        elif any(word in p for word in ["habit", "track", "consistency", "routine"]):
            return "behavioral_coaching"
        elif any(word in p for word in ["purpose", "mission", "values", "north star"]):
            return "purpose_alignment"
        else:
            return "general_chat"

    def _select_model(self, task_type: str):
        """
        Maps task type to preferred model according to Neuraline’s orchestration matrix.
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