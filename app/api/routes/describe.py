from fastapi import APIRouter, UploadFile, File
from app.utils.image import validate_image, bytes_to_pil
from app.services.descriptor import describe_image
from app.models.schemas import DescriptionResponse

router = APIRouter()


@router.post("/describe", response_model=DescriptionResponse)
async def describe(file: UploadFile = File(...)):
    image_bytes = await validate_image(file)
    image = bytes_to_pil(image_bytes)
    result, _ = describe_image(image)
    return result
