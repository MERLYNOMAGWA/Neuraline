import asyncio
import functools
import logging

logger = logging.getLogger(__name__)

def retry_async(retries: int = 2, delay: float = 0.5):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Retry {i+1}/{retries} for {fn.__name__} failed: {e}")
                    if i < retries:
                        await asyncio.sleep(delay * (2 ** i))
            raise last_exc
        return wrapper
    return deco
