from typing import Any, Optional
import json
from redis import asyncio as aioredis
from datetime import timedelta

class SessionCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis = aioredis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _get_session_key(self, app_id: str, user_id: str, session_id: str) -> str:
        return f"session:{app_id}:{user_id}:{session_id}"

    def _get_messages_key(self, app_id: str, user_id: str, session_id: str) -> str:
        return f"messages:{app_id}:{user_id}:{session_id}"

    async def get_session(self, app_id: str, user_id: str, session_id: str) -> Optional[dict]:
        """Get session data from cache"""
        key = self._get_session_key(app_id, user_id, session_id)
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set_session(self, app_id: str, user_id: str, session_id: str, data: dict, ttl: Optional[int] = None) -> None:
        """Set session data in cache"""
        key = self._get_session_key(app_id, user_id, session_id)
        await self.redis.set(key, json.dumps(data), ex=ttl or self.default_ttl)

    async def get_messages(self, app_id: str, user_id: str, session_id: str) -> Optional[list]:
        """Get session messages from cache"""
        key = self._get_messages_key(app_id, user_id, session_id)
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set_messages(self, app_id: str, user_id: str, session_id: str, messages: list, ttl: Optional[int] = None) -> None:
        """Set session messages in cache"""
        key = self._get_messages_key(app_id, user_id, session_id)
        await self.redis.set(key, json.dumps(messages), ex=ttl or self.default_ttl)

    async def invalidate_session(self, app_id: str, user_id: str, session_id: str) -> None:
        """Invalidate session and its messages from cache"""
        session_key = self._get_session_key(app_id, user_id, session_id)
        messages_key = self._get_messages_key(app_id, user_id, session_id)
        await self.redis.delete(session_key, messages_key)

    async def close(self) -> None:
        """Close Redis connection"""
        await self.redis.close() 