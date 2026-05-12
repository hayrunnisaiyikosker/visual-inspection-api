from fastapi import APIRouter, UploadFile, File
from app.utils.image import validate_image, bytes_to_pil
from app.services.bg_remover import remove_background
from app.models.schemas import BackgroundRemovalResponse

router = APIRouter()


@router.post("/remove-background", response_model=BackgroundRemovalResponse)
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await validate_image(file)
    image = bytes_to_pil(image_bytes)
    return remove_background(image)
