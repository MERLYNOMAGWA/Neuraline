import asyncio
import logging
from typing import Dict

from langchain.memory import ConversationBufferMemory
from app.services.retriever import ContextRetriever
from app.services.ai_clients import GeminiClient, GroqClient
from app.services.model_router import ModelRouter
from app.prompts.templates import (
    reflection_prompt,
    reasoning_prompt,
    coaching_prompt,
    purpose_prompt,
    general_prompt,
)
from app.core.logging_config import log_event
from app.services.memory.chroma_memory import ChromaConversationMemory
from app.services.safety.content_filter import ContentFilter
from app.services.safety.response_validator import ResponseValidator

logger = logging.getLogger(__name__)


class ConversationManager:
    """Handles multi-turn chat sessions with persistence, routing, and safety."""

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()
        self.model_router = ModelRouter()
        self.retriever = ContextRetriever()
        self.memory_store = ChromaConversationMemory()
        self.filter = ContentFilter()
        self.validator = ResponseValidator()
        self._memories: Dict[str, ConversationBufferMemory] = {}

    def _get_memory(self, session_id: str):
        """Get or create conversation memory for a given session."""
        from langchain.memory import ConversationBufferMemory
        from langchain_core.messages import HumanMessage, AIMessage

        mem = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
        )

        if session_id not in self._memories:
            try:
                persisted = self.memory_store.load_session_history(session_id)
                if persisted:
                    messages = []
                    for msg in persisted:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))
                    mem.chat_memory.messages = messages
                    logger.info(f"[MEMORY_LOAD] Loaded memory for session={session_id}")
            except Exception as e:
                logger.warning(f"[MEMORY_LOAD_ERROR] Failed to load memory for session={session_id}: {e}")

            self._memories[session_id] = mem

        return self._memories[session_id]

    def _classify_task(self, prompt: str) -> str:
        """Simple heuristic task classifier."""
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
        session_id: str,
        user_id: str | None = None,
        use_router: bool = True,
    ) -> str:
        """Main conversational entrypoint."""
        user_input = self.filter.clean(user_input)
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
                await self.model_router.run(final_prompt, task_type=task_type)
                if use_router
                else await self.gemini.generate(final_prompt)
            )

            response = self.validator.clean(response)

            try:
                memory.save_context({"input": user_input}, {"output": response})
                if hasattr(self.memory_store, "save_message"):
                    self.memory_store.save_message(session_id, "user", user_input)
                    self.memory_store.save_message(session_id, "assistant", response)
                log_event("MEMORY_SAVE", f"Persisted chat turn for session={session_id}")
            except Exception as e:
                log_event("MEMORY_SAVE_ERROR", f"⚠️ Failed to persist memory: {e}")

            return response
        except Exception as e:
            log_event("CONVERSATION_ERROR", f"❌ Conversation error: {e}")
            logger.exception("Conversation error")
            raise