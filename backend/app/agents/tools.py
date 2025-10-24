import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def send_reminder_webhook(payload: Dict[str, Any], webhook_url: str) -> Dict[str, Any]:
    logger.info(f"Tool: send_reminder_webhook -> {webhook_url} payload keys {list(payload.keys())}")
    await asyncio.sleep(0.05)
    return {"ok": True, "webhook": webhook_url}

async def schedule_task(payload: Dict[str, Any], when: str) -> Dict[str, Any]:
    logger.info(f"Tool: schedule_task -> when={when} payload keys {list(payload.keys())}")
    await asyncio.sleep(0.05)
    return {"ok": True, "scheduled_for": when}
