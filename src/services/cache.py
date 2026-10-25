import hashlib
import json
import logging
from typing import Optional, Dict, Any
import redis.asyncio as aioredis

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheService:
    """Async Redis query cache with graceful fallback."""

    def __init__(self, redis_url: str, ttl_seconds: int = 3600, enabled: bool = True):
        self.redis_url = redis_url
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._client: Optional[aioredis.Redis] = None

    async def get_client(self) -> Optional[aioredis.Redis]:
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                )
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
                return None
        return self._client

    def _hash_key(self, query: str) -> str:
        norm = query.strip().lower().encode("utf-8")
        return f"rag:query:{hashlib.sha256(norm).hexdigest()}"

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            client = await self.get_client()
            if client is None:
                return None
            cached_data = await client.get(self._hash_key(query))
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Redis cache get error: {e}")
        return None

    async def set(self, query: str, data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            client = await self.get_client()
            if client is None:
                return
            await client.setex(
                name=self._hash_key(query),
                time=self.ttl,
                value=json.dumps(data),
            )
        except Exception as e:
            logger.debug(f"Redis cache set error: {e}")

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


_cache_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService(
            redis_url=settings.REDIS_URL,
            ttl_seconds=settings.CACHE_TTL_SECONDS,
            enabled=settings.CACHE_ENABLED,
        )
    return _cache_instance
