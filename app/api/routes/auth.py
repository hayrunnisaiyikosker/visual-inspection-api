from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.api.deps import get_db_session
from app.models.database import ApiKey
from pydantic import BaseModel
from datetime import datetime, timedelta
import hashlib
import secrets
import uuid

router = APIRouter()

class CreateApiKeyRequest(BaseModel):
    name: str
    monthly_quota: int = 1000

class CreateApiKeyResponse(BaseModel):
    api_key: str
    name: str
    monthly_quota: int
    message: str

@router.post("/keys", response_model=CreateApiKeyResponse)
async def create_api_key(
    request: CreateApiKeyRequest,
    db: AsyncSession = Depends(get_db_session),
):
    raw_key = "vi-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    existing = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Key collision, try again")

    db_key = ApiKey(
        id=uuid.uuid4(),
        key_hash=key_hash,
        name=request.name,
        monthly_quota=request.monthly_quota,
        requests_used=0,
        quota_reset_at=datetime.utcnow() + timedelta(days=30),
        is_active=True,
    )
    db.add(db_key)
    await db.commit()

    return CreateApiKeyResponse(
        api_key=raw_key,
        name=request.name,
        monthly_quota=request.monthly_quota,
        message="Store this key safely. It will not be shown again.",
    )

@router.get("/keys")
async def list_api_keys(db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(ApiKey))
    keys = result.scalars().all()
    return [
        {
            "id": str(k.id),
            "name": k.name,
            "monthly_quota": k.monthly_quota,
            "requests_used": k.requests_used,
            "is_active": k.is_active,
            "quota_reset_at": str(k.quota_reset_at),
        }
        for k in keys
    ]
