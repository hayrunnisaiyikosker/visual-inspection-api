from fastapi import APIRouter, UploadFile, File
from app.utils.image import validate_image, bytes_to_pil
from app.services.classifier import classify_image
from app.services.descriptor import describe_image
from app.services.detector import detect_objects
from app.services.bg_remover import remove_background
from app.models.schemas import UnifiedAnalysisResponse, ProcessingTime
import time

router = APIRouter()


@router.post("/", response_model=UnifiedAnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    image_bytes = await validate_image(file)
    image = bytes_to_pil(image_bytes)

    total_start = time.time()

    # Classification
    t0 = time.time()
    classification = classify_image(image)
    classification_ms = round((time.time() - t0) * 1000, 2)

    # Description
    t0 = time.time()
    description = describe_image(image)
    description_ms = round((time.time() - t0) * 1000, 2)

    # Detection
    t0 = time.time()
    detection = detect_objects(image)
    detection_ms = round((time.time() - t0) * 1000, 2)

    # Background removal
    t0 = time.time()
    bg_result = remove_background(image)
    bg_ms = round((time.time() - t0) * 1000, 2)

    total_ms = round((time.time() - total_start) * 1000, 2)

    return UnifiedAnalysisResponse(
        classification=classification,
        description=description,
        detection=detection,
        background_removed=bg_result.image_base64,
        processing_time_ms=ProcessingTime(
            classification_ms=classification_ms,
            description_ms=description_ms,
            detection_ms=detection_ms,
            background_removal_ms=bg_ms,
            total_ms=total_ms,
        ),
    )
