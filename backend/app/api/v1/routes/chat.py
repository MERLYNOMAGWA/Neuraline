from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.mcp.mcp_engine import MCPEngine

router = APIRouter()
mcp_engine = MCPEngine()

class ChatRequest(BaseModel):
    message: str
    session_id: str
    mode: Optional[str] = "chain"
    roles: Optional[List[str]] = None
    timeout: Optional[int] = None

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
# Run the MCP engine chain (multi-agent reasoning)
        result = await mcp_engine.run(
        query=request.message,
        session_id=request.session_id,
        mode=request.mode or "chain",
        roles=request.roles,
        timeout=request.timeout,
    )

    # Retrieve each agent’s snapshot output
        snapshot = result.get("snapshot", {})
        combined_output = result.get("combined", "")
        best_role = result.get("best_role", "reflector")

    # --- Fusion Step ---
    # Smoothly merge all agent perspectives into a unified Neuraline voice
        fused_text = mcp_engine._fuse_dialogue(snapshot)

        if not fused_text.strip():
            fused_text = "I'm here with you. How are you feeling right now?"

        neuraline_voice = f"Hey there 👋 — {fused_text.strip()}"

        return {
            "sender": "Neuraline",
            "reply": neuraline_voice,
            "best_role": best_role,
            "mode": result.get("mode", "chain"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))