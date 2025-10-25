from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routes import health, auth, mcp, chat
from app.core.config import settings
from app.core.logging_config import log_event

app = FastAPI(title=settings.project_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=f"{settings.api_v1_str}")
app.include_router(auth.router, prefix=f"{settings.api_v1_str}")
app.include_router(mcp.router, prefix="/api/v1/mcp", tags=["mcp"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.on_event("startup")
async def startup_event():
    log_event("STARTUP", "🚀 Neuraline backend started successfully")

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Neuraline API"}