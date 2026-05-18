from fastapi import APIRouter, UploadFile, File
from app.utils.image import validate_image, bytes_to_pil
from app.services.detector import detect_objects
from app.models.schemas import DetectionResponse

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image_bytes = await validate_image(file)
    image = bytes_to_pil(image_bytes)
    result, _ = detect_objects(image)
    return result
