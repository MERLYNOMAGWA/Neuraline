import asyncio
from typing import Any, Dict

class Blackboard:
    """Simple async-safe blackboard for cross-agent shared state."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._store: Dict[str, Any] = {}

    async def read(self, key: str, default=None):
        async with self._lock:
            return self._store.get(key, default)

    async def write(self, key: str, value):
        async with self._lock:
            self._store[key] = value

    async def update_dict(self, key: str, values: Dict):
        """If the value at key is a dict, update it, else set it."""
        async with self._lock:
            cur = self._store.get(key, {})
            if not isinstance(cur, dict):
                cur = {}
            cur.update(values)
            self._store[key] = cur

    async def dump(self):
        async with self._lock:
            return dict(self._store)

    async def clear(self):
        async with self._lock:
            self._store.clear()