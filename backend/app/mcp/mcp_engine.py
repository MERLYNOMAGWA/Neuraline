import asyncio
import logging
from typing import Dict, List, Optional, Any

from app.services.model_router import ModelRouter
from app.services.retriever import ContextRetriever
from app.services.memory.chroma_memory import ChromaConversationMemory
from app.core.logging_config import log_event

logger = logging.getLogger(__name__)

AGENT_PROFILES: Dict[str, str] = {
    "reflector": (
        "Reflection Agent: empathetic, asks reflective questions, surfaces emotions and "
        "internal drivers. Helps the user observe without judgement."
    ),
    "strategist": (
        "Strategist Agent: structured reasoning, breaks goals into steps and weekly plans. "
        "Produces clear, actionable milestones."
    ),
    "coach": (
        "Consistency Coach: behavior-focused, creates micro-habits, nudges and accountability "
        "structures to sustain actions."
    ),
    "purpose": (
        "Purpose Agent: connects tasks and actions to deeper values, purpose and long-term "
        "alignment. Produces short mission statements and motivation framing."
    ),
}

class MCPEngine:
    """
    Model Context Protocol engine for coordinating multiple agents.

    - Shares RAG context and session memory
    - Supports 'parallel' or 'chain' execution modes
    - Provides graceful fallback fusion into Neuraline voice
    """

    def __init__(
        self,
        retriever: Optional[ContextRetriever] = None,
        model_router: Optional[ModelRouter] = None,
        memory_store: Optional[ChromaConversationMemory] = None,
    ):
        self.retriever = retriever or ContextRetriever()
        self.model_router = model_router or ModelRouter()
        self.memory_store = memory_store or ChromaConversationMemory()
        self.agent_timeout = 30
        self.retries = 1

    async def _get_context(self, query: str) -> str:
        try:
            ctx = await asyncio.to_thread(self.retriever.retrieve, query)
            return ctx or ""
        except Exception as e:
            logger.warning("MCP: retrieval failed: %s", e)
            return ""

    def _build_agent_prompt(
        self, role: str, context: str, query: str, snapshot: Optional[Dict[str, str]] = None
    ) -> str:
        profile = AGENT_PROFILES.get(role, f"{role} agent")
        snapshot_text = ""
        if snapshot:
            snapshot_text = "\n\nPrevious agent snapshots:\n" + "\n".join(
                f"[{r}] {t}" for r, t in snapshot.items()
            )
        prompt = (
            f"[{role.upper()} AGENT]\n"
            f"Role description:\n{profile}\n\n"
            f"Context:\n{context or 'No context available.'}\n\n"
            f"User query:\n{query}\n\n"
            f"{snapshot_text}\n\n"
            f"Please respond concisely and include helpful next steps or reflective questions where relevant."
        )
        return prompt

    def _fuse_dialogue(self, snapshot: dict) -> str:
        """
        Combine multiple agent outputs into one emotionally aware Neuraline-style message.
        Safe for async FastAPI contexts (no event-loop conflicts).
        """
        try:
            parts = []
            for role, text in snapshot.items():
                if not text or not isinstance(text, str):
                    continue
                if role == "reflector":
                    parts.append(f"It sounds like you're reflecting deeply. {text.strip()}")
                elif role == "strategist":
                    parts.append(f"Here's a practical step forward: {text.strip()}")
                elif role == "coach":
                    parts.append(f"To keep it consistent, {text.strip()}")
                elif role == "purpose":
                    parts.append(f"And don't forget why this matters — {text.strip()}")
                else:
                    parts.append(text.strip())

            if not parts:
                return "I'm here with you. How are you feeling right now?"

            fused_text = " ".join(parts)
            fused_text = fused_text.replace("\n", " ").replace("  ", " ").strip()
            return fused_text

        except Exception as e:
            return (
                f"It sounds like a lot is happening — but I'm here to help you slow down and find clarity. "
                f"(fusion fallback: {e})"
            )

    async def _call_agent(self, role: str, prompt: str) -> Dict[str, Any]:
        """
        Calls the model router with the agent prompt and returns a result dict.
        Handles retries and provides fallback text on failure.
        """
        attempt = 0
        last_exc = None
        while attempt <= self.retries:
            try:
                response = await asyncio.wait_for(
                    self.model_router.run(prompt), timeout=self.agent_timeout
                )
                return {"role": role, "success": True, "output": response}
            except Exception as e:
                last_exc = e
                attempt += 1
                logger.warning("MCP: agent %s call failed (attempt %s): %s", role, attempt, e)
                await asyncio.sleep(0.5 * attempt)

        fallback_msg = (
            f"(local fallback by {role}) The {role} agent could not produce a response right now."
        )
        return {"role": role, "success": False, "error": str(last_exc), "output": fallback_msg}

    async def run(
        self,
        query: str,
        session_id: str,
        mode: str = "chain",
        roles: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run MCP orchestration with reflection → strategy → coaching → purpose fusion.
        """
        roles = roles or ["reflector", "strategist", "coach", "purpose"]
        if timeout:
            self.agent_timeout = timeout

        context = await self._get_context(query)
        log_event("MCP", f"MCP run start session={session_id} mode={mode} roles={roles}")

        try:
            persisted = await asyncio.to_thread(self.memory_store.load_session_history, session_id)
        except Exception as e:
            logger.debug("MCP: failed to load session memory: %s", e)
            persisted = []

        memory_text = ""
        if persisted:
            memory_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in persisted[-10:])

        results: Dict[str, Dict[str, Any]] = {}
        snapshot: Dict[str, str] = {}

        if mode == "parallel":
            tasks = []
            for role in roles:
                prompt = self._build_agent_prompt(role, context or memory_text, query, snapshot=None)
                tasks.append(self._call_agent(role, prompt))
            agent_outputs = await asyncio.gather(*tasks)
            for res in agent_outputs:
                results[res["role"]] = res
                snapshot[res["role"]] = res.get("output", "")

            best_role = next((r for r in roles if results.get(r, {}).get("success")), roles[0])
            combined = self._fuse_dialogue(snapshot)
            log_event("MCP_PARALLEL", f"session={session_id} best_role={best_role}")

            return {
                "mode": "parallel",
                "best_role": best_role,
                "snapshot": snapshot,
                "combined": combined,
                "results": results,
            }

        else: 
            for role in roles:
                prompt = self._build_agent_prompt(role, context or memory_text, query, snapshot=snapshot)
                res = await self._call_agent(role, prompt)
                results[role] = res
                snapshot[role] = res.get("output", "")

            best_role = next((r for r in roles if results.get(r, {}).get("success")), roles[0])
            combined = self._fuse_dialogue(snapshot)
            log_event("MCP_CHAIN", f"session={session_id} best_role={best_role}")

            return {
                "mode": "chain",
                "best_role": best_role,
                "snapshot": snapshot,
                "combined": combined,
                "results": results,
            }

    async def run_mcp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper for unified MCP execution — accepts dict, forwards to run().
        """
        query = request.get("query")
        session_id = request.get("session_id", "anonymous")
        mode = request.get("mode", "chain")
        roles = request.get("roles", None)
        timeout = request.get("timeout", None)

        return await self.run(query=query, session_id=session_id, mode=mode, roles=roles, timeout=timeout)