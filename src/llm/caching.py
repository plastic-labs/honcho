from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from pydantic import BaseModel

from src.config import ModelConfig, PromptCachePolicy

__all__ = [
    "GeminiCacheHandle",
    "InMemoryGeminiCacheStore",
    "PromptCachePolicy",
    "build_cache_key",
    "gemini_cache_store",
]


class GeminiCacheHandle(BaseModel):
    key: str
    cached_content_name: str
    expires_at: datetime


def build_cache_key(
    *,
    config: ModelConfig,
    cache_policy: PromptCachePolicy,
    cacheable_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    payload = {
        "transport": config.transport,
        "model": config.model,
        "cache_policy": cache_policy.model_dump(mode="json"),
        "messages": cacheable_messages,
        "tools": tools,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"llm-cache:{cache_policy.key_version}:{digest}"


class InMemoryGeminiCacheStore:
    """Best-effort local cache for Gemini cached-content handles.

    Uses LRU eviction with a max entry limit to prevent unbounded growth.
    """

    MAX_ENTRIES: int = 1024

    def __init__(self) -> None:
        self._handles: OrderedDict[str, GeminiCacheHandle] = OrderedDict()
        self._lock: Lock = Lock()

    def get(self, key: str) -> GeminiCacheHandle | None:
        with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                return None
            if handle.expires_at <= datetime.now(timezone.utc):
                self._handles.pop(key, None)
                return None
            self._handles.move_to_end(key)
            return handle

    def set(self, handle: GeminiCacheHandle) -> GeminiCacheHandle:
        with self._lock:
            now = datetime.now(timezone.utc)
            expired = [k for k, h in self._handles.items() if h.expires_at <= now]
            for k in expired:
                self._handles.pop(k, None)
            if handle.key in self._handles:
                self._handles.move_to_end(handle.key)
            self._handles[handle.key] = handle
            while len(self._handles) > self.MAX_ENTRIES:
                self._handles.popitem(last=False)
        return handle


gemini_cache_store = InMemoryGeminiCacheStore()
