import redis.asyncio as aioredis
import json
from app.core.config import settings

redis_client = None


async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return redis_client


async def get_cached_result(image_hash: str):
    r = await get_redis()
    data = await r.get(f"result:{image_hash}")
    if data:
        return json.loads(data)
    return None


async def set_cached_result(image_hash: str, result: dict):
    r = await get_redis()
    await r.setex(
        f"result:{image_hash}",
        settings.CACHE_TTL,
        json.dumps(result),
    )
