from fastapi import APIRouter
from . import health, auth, mcp, chat

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])