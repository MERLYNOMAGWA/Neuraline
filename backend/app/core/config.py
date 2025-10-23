from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    #project info
    project_name: str = Field("neuraline", env="PROJECT_NAME")
    api_v1_str: str = Field("/api/v1", env="API_V1_STR")
    #security
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    #environment
    environment: str = Field("development", env="ENVIRONMENT")

    #langSmith
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")
    langsmith_tracing: bool | None = Field(False, env="LANGSMITH_TRACING")
    langsmith_endpoint: str | None = Field("https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    langsmith_project: str | None = Field("neuraline", env="LANGSMITH_PROJECT")

    #Models
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")
    groq_api_key: str | None = Field(None, env="GROQ_API_KEY")
    hf_token: str | None = Field(None, env="HF_TOKEN")
    google_api_key: str | None = Field(None, env="GOOGLE_API_KEY")

    #vector DB
    chroma_path: str = Field("./chroma_storage", env="CHROMA_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  
settings = Settings()