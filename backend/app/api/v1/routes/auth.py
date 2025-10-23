from fastapi import APIRouter, HTTPException, status
from app.schemas.auth import LoginRequest, TokenResponse
from app.core.security import create_access_token

router = APIRouter()

@router.post("/login", response_model=TokenResponse, tags=["Auth"])
async def login(payload: LoginRequest):
    if payload.username == "demo" and payload.password == "password":
        token = create_access_token({"sub": payload.username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")