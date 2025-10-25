from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from app.mcp.mcp_engine import MCPEngine

router = APIRouter(prefix="/api/v1/mcp", tags=["MCP"])
logger = logging.getLogger(__name__)

mcp_engine = MCPEngine()


class MCPRequest(BaseModel):
    query: str
    session_id: str
    mode: Optional[str] = "chain" 
    roles: Optional[List[str]] = None
    timeout: Optional[int] = None


@router.post("/run")
async def run_mcp(req: MCPRequest) -> Dict[str, Any]:
    """
    Executes the Model Context Protocol (MCP) orchestration pipeline.
    This endpoint coordinates multiple cognitive agents (reflector, strategist, coach, purpose)
    to produce a structured, human-like multi-perspective response.
    """
    try:
        if hasattr(mcp_engine, "run_mcp"):
            result = await mcp_engine.run_mcp(req.dict())
        else:
            result = await mcp_engine.run(
                query=req.query,
                session_id=req.session_id,
                mode=req.mode or "chain",
                roles=req.roles,
                timeout=req.timeout,
            )

        return {
            "ok": True,
            "session_id": req.session_id,
            "mode": result.get("mode"),
            "best_role": result.get("best_role"),
            "snapshot": result.get("snapshot"),
            "combined": result.get("combined"),
            "results": result.get("results"),
        }

    except Exception as e:
        logger.exception(f"MCP execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"MCP internal error: {str(e)}")
