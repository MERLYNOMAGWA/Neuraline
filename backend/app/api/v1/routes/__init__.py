from fastapi import APIRouter
from . import health, auth, ai_test 

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(ai_test.router, prefix="/ai", tags=["ai"])