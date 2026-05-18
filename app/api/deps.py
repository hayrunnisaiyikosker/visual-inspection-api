from app.models.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from typing import AsyncGenerator


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session
