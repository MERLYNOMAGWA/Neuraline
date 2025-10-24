import asyncio
import logging
from typing import Dict

from langchain.memory import ConversationBufferMemory

from app.services.retriever import ContextRetriever
from app.services.ai_clients import GeminiClient, GroqClient
from app.core.logging_config import log_event
from app.services.model_router import ModelRouter
from app.prompts.templates import (
    reflection_prompt,
    reasoning_prompt,
    coaching_prompt,
    purpose_prompt,
    general_prompt,
)

logger = logging.getLogger(__name__)


class ConversationManager:
    """Handles multi-turn chat sessions with context retrieval and memory."""

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()
        self.model_router = ModelRouter()

        self.retriever = ContextRetriever()

        self._memories: Dict[str, ConversationBufferMemory] = {}

    def _get_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for the session."""
        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=False,
            )
        return self._memories[session_id]

    def _classify_task(self, prompt: str) -> str:
        """Simple keyword-based task classifier."""
        p = prompt.lower()
        if any(w in p for w in ["reflect", "feeling", "emotion", "journal"]):
            return "emotional_reflection"
        if any(w in p for w in ["plan", "goal", "steps", "strategy", "how to achieve"]):
            return "cognitive_reasoning"
        if any(w in p for w in ["summarize", "analyze", "context", "rag", "retrieve", "neuraline"]):
            return "rag_query"
        if any(w in p for w in ["habit", "track", "consistency", "routine"]):
            return "behavioral_coaching"
        if any(w in p for w in ["purpose", "mission", "values", "north star"]):
            return "purpose_alignment"
        return "general_chat"

    def _select_template(self, task_type: str):
        """Map task type to the correct prompt template."""
        mapping = {
            "emotional_reflection": reflection_prompt,
            "cognitive_reasoning": reasoning_prompt,
            "rag_query": reasoning_prompt,
            "behavioral_coaching": coaching_prompt,
            "purpose_alignment": purpose_prompt,
            "general_chat": general_prompt,
        }
        return mapping.get(task_type, general_prompt)

    async def chat(
        self,
        user_input: str,
        session_id: str = "default_session",
        use_router: bool = True,
    ) -> str:
        """
        Main entrypoint for multi-turn chat.
        Handles classification, retrieval, memory, and model routing.
        """
        task_type = self._classify_task(user_input)
        log_event("CONVERSATION", f"Incoming message classified as {task_type}")

        context = ""
        if task_type in (
            "rag_query",
            "emotional_reflection",
            "cognitive_reasoning",
            "purpose_alignment",
        ):
            try:
                log_event("RAG_RETRIEVE", f"🔍 Retrieving context for session={session_id}")
                context = await asyncio.to_thread(self.retriever.retrieve, user_input)
                log_event("RAG_CONTEXT", f"Retrieved {len(context)} chars for session={session_id}")
            except Exception as e:
                log_event("RAG_ERROR", f"⚠️ Retrieval failed: {e}")
                context = ""

        memory = self._get_memory(session_id)
        memory_text = getattr(memory, "buffer", "")

        template = self._select_template(task_type)
        prompt_text = template.format(
            context=context or "No context available.",
            memory=memory_text,
            user_input=user_input,
        )
        final_prompt = (
            "You are Neuraline — an empathetic, purpose-driven cognitive companion.\n\n"
            + prompt_text
        )
        log_event("PROMPT", f"Prepared prompt for session={session_id}, task={task_type}")

        try:
            response = (
                await self.model_router.run(final_prompt)
                if use_router
                else await self.gemini.generate(final_prompt)
            )

            try:
                memory.save_context({"input": user_input}, {"output": response})
            except Exception:
                if hasattr(memory, "buffer"):
                    memory.buffer = (memory.buffer or "") + f"\nUser: {user_input}\nAI: {response}"
            log_event("MEMORY", f"Saved conversation turn for session={session_id}")

            return response
        except Exception as e:
            log_event("CONVERSATION_ERROR", f"❌ Conversation error: {e}")
            logger.exception("Conversation error")
            raise