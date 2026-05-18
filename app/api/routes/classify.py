from fastapi import APIRouter, UploadFile, File
from app.utils.image import validate_image, bytes_to_pil
from app.services.classifier import classify_image
from app.models.schemas import ClassificationResponse

router = APIRouter()


@router.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    image_bytes = await validate_image(file)
    image = bytes_to_pil(image_bytes)
    result, _ = classify_image(image)
    return result
