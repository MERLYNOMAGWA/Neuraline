import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents. Agents should implement `run`."""

    name = "base"

    def __init__(self, retriever=None, model_router=None, memory_store=None):
        self.retriever = retriever
        self.model_router = model_router
        self.memory_store = memory_store

    async def run(self, query: str, session_id: Optional[str], blackboard) -> Dict[str, Any]:
        """Run the agent. Return a dict with at least 'role' and 'output'."""
        raise NotImplementedError()

    async def _call_model(self, prompt: str, task_type: str = "general_chat") -> str:
        """Helper to call the model router if available, else do a simple fallback."""
        if self.model_router:
            try:
                return await self.model_router.run(prompt, task_type=task_type)
            except Exception as e:
                logger.warning(f"{self.name} model_router failed: {e}")
        return f"(local fallback by {self.name}) {prompt[:300]}"
