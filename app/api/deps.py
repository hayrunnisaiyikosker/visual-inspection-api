from app.models.database import get_db, ApiKey
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from typing import AsyncGenerator
from app.services.cache import get_redis
import hashlib

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session

async def verify_api_key(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db_session),
) -> ApiKey:
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header missing")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
    )
    db_key = result.scalar_one_or_none()

    if not db_key:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if db_key.requests_used >= db_key.monthly_quota:
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")

    try:
        r = await get_redis()
        rate_key = f"rate:{str(db_key.id)}"
        current = await r.get(rate_key)
        if current and int(current) >= 100:
            raise HTTPException(status_code=429, detail="Rate limit exceeded: max 100 requests per hour")
        pipe = r.pipeline()
        await pipe.incr(rate_key)
        await pipe.expire(rate_key, 3600)
        await pipe.execute()
    except HTTPException:
        raise
    except Exception:
        pass

    return db_key
