import logging
from datetime import datetime
from langsmith import Client
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuraline")

try:
    client = Client(api_key=settings.langsmith_api_key)
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize LangSmith client: {e}")
    client = None


def log_event(event_type: str, message: str, level: str = "info", run_type: str = "tool"):
    """
    Logs an event locally and optionally sends it to LangSmith.

    Args:
        event_type (str): short string (e.g., 'startup', 'request', 'error', 'model_call')
        message (str): human-readable description
        level (str): log level ('info', 'warning', 'error', 'debug')
        run_type (str): LangSmith run type ('tool', 'chain', 'llm', 'retriever', etc.)
    """
    timestamp = datetime.utcnow().isoformat()

    log_method = getattr(logger, level, logger.info)
    log_method(f"[{event_type.upper()}] {message}")

    if client:
        try:
            client.create_run(
                name=event_type,
                run_type=run_type,
                inputs={"message": message, "timestamp": timestamp},
                project_name=settings.langsmith_project,
            )
        except Exception as e:
            logger.warning(f"Langsmith log failed: {e}")