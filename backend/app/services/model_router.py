import logging
from app.services.ai_clients import GeminiClient, GroqClient
from app.core.logging_config import log_event

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes requests intelligently between models with fallback."""

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()

    async def run(self, prompt: str) -> str:
        try:
            log_event("model_call", "🧠 Routing request to Gemini", level="info", run_type="llm")
            return await self.gemini.generate(prompt)

        except Exception as e:
            log_event("error", f"⚠️ Gemini failed — switching to Groq ({e})", level="warning", run_type="llm")

            try:
                result = await self.groq.generate(prompt)
                log_event("model_call", "✅ Groq successfully handled fallback request", level="info", run_type="llm")
                return result

            except Exception as e2:
                log_event("error", f"❌ Both Gemini and Groq failed: {e2}", level="error", run_type="llm")
                logger.exception("All model routes failed.")
                raise RuntimeError("All model routes failed.") from e2