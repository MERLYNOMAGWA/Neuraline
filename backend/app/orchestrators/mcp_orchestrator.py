import asyncio
import logging
from typing import Dict, Any

from app.mcp.mcp_engine import MCPEngine

logger = logging.getLogger(__name__)

mcp_engine = MCPEngine()

class MCPOrchestrator:
    """
    Orchestrator layer connecting FastAPI endpoints to the MCP Engine.
    Handles async execution, structured responses, and error safety.
    """

    def __init__(self):
        self.engine = mcp_engine

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely processes a client request using the MCP engine.
        """
        try:
            logger.info(f"[MCP_ORCH] Running MCP for session {request_data.get('session_id')}")
            result = await self.engine.run_mcp(request_data)
            return {
                "status": "success",
                "session_id": request_data.get("session_id"),
                "mode": result.get("mode"),
                "best_role": result.get("best_role"),
                "snapshot": result.get("snapshot"),
                "response": result.get("combined"),
            }
        except Exception as e:
            logger.error(f"MCP orchestrator error: {e}", exc_info=True)
            return {
                "status": "error",
                "detail": str(e),
            }