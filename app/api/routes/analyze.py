from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.image import validate_image, bytes_to_pil, compute_image_hash
from app.services.classifier import classify_image
from app.services.descriptor import describe_image
from app.services.detector import detect_objects
from app.services.bg_remover import remove_background
from app.services.cache import get_cached_result, set_cached_result
from app.models.schemas import UnifiedAnalysisResponse, ProcessingTime
from app.models.database import Request, InferenceResult, ApiKey, get_db
from app.api.deps import verify_api_key, get_db_session
import asyncio
import gc
import time
import uuid

router = APIRouter()

@router.post("/", response_model=UnifiedAnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    api_key: ApiKey = Depends(verify_api_key),
):
    image_bytes = await validate_image(file)
    image_hash = compute_image_hash(image_bytes)
    try:
        cached = await get_cached_result(image_hash)
        if cached:
            return UnifiedAnalysisResponse(**cached)
    except Exception:
        pass

    image = bytes_to_pil(image_bytes)
    loop = asyncio.get_event_loop()
    total_start = time.time()

    # Modeller SIRAYLA çalışır — aynı anda değil
    classification_result, classify_ms = await loop.run_in_executor(None, classify_image, image)
    gc.collect()
    description_result, describe_ms = await loop.run_in_executor(None, describe_image, image)
    gc.collect()
    detection_result, detect_ms = await loop.run_in_executor(None, detect_objects, image, classification_result.top_prediction)
    gc.collect()
    bg_result, bg_ms = await loop.run_in_executor(None, remove_background, image)
    gc.collect()

    total_ms = round((time.time() - total_start) * 1000, 2)
    processing_time = ProcessingTime(
        classification_ms=classify_ms,
        description_ms=describe_ms,
        detection_ms=detect_ms,
        background_removal_ms=bg_ms,
        total_ms=total_ms,
    )
    response = UnifiedAnalysisResponse(
        classification=classification_result,
        description=description_result,
        detection=detection_result,
        background_removed=bg_result.image_base64,
        processing_time_ms=processing_time,
    )
    try:
        request_id = uuid.uuid4()
        db_request = Request(
            id=request_id,
            api_key_id=api_key.id,
            image_hash=image_hash,
            image_path=file.filename or "upload",
            total_processing_ms=total_ms,
            status="completed",
        )
        db.add(db_request)
        await db.flush()
        db_result = InferenceResult(
            id=uuid.uuid4(),
            request_id=request_id,
            classification=classification_result.model_dump(),
            description=description_result.model_dump(),
            detected_objects=[obj.model_dump() for obj in detection_result.detected_objects],
            background_removed_b64=bg_result.image_base64,
            classify_ms=classify_ms,
            describe_ms=describe_ms,
            detect_ms=detect_ms,
            bg_remove_ms=bg_ms,
        )
        db.add(db_result)
        api_key.requests_used += 1
        await db.commit()
        print(f"[DB] Saved request {request_id}")
    except Exception as e:
        await db.rollback()
        print(f"[DB] Write failed: {e}")
    try:
        await set_cached_result(image_hash, response.model_dump())
    except Exception:
        pass
    return response
