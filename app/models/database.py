from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    Text, TIMESTAMP, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from app.core.config import settings

engine = create_async_engine(settings.async_database_url, echo=False)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    monthly_quota = Column(Integer, nullable=False, default=1000)
    requests_used = Column(Integer, nullable=False, default=0)
    quota_reset_at = Column(TIMESTAMP, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    requests = relationship("Request", back_populates="api_key")


class Request(Base):
    __tablename__ = "requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)
    image_hash = Column(String(64), nullable=False)
    image_path = Column(Text, nullable=False)
    total_processing_ms = Column(Float)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    api_key = relationship("ApiKey", back_populates="requests")
    inference_result = relationship("InferenceResult", back_populates="request", uselist=False)


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("requests.id"), unique=True, nullable=False)
    classification = Column(JSONB)
    description = Column(JSONB)
    detected_objects = Column(JSONB)
    background_removed_b64 = Column(Text)
    classify_ms = Column(Float)
    describe_ms = Column(Float)
    detect_ms = Column(Float)
    bg_remove_ms = Column(Float)

    request = relationship("Request", back_populates="inference_result")


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
