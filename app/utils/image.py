from fastapi import HTTPException, UploadFile
from PIL import Image
import io
import hashlib
from app.core.config import settings


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


async def validate_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WebP",
        )

    contents = await file.read()

    if len(contents) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents) / 1024 / 1024:.1f} MB. Maximum: {settings.MAX_FILE_SIZE_MB} MB",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid image file. Please upload a valid JPEG, PNG, or WebP.",
        )

    image = Image.open(io.BytesIO(contents))
    width, height = image.size

    if width < 50 or height < 50:
        raise HTTPException(
            status_code=422,
            detail=f"Image too small: {width}x{height}px. Minimum: 50x50px",
        )

    return contents


def compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    # Her zaman RGB'ye çevir — model RGBA, L, P gibi formatları kabul etmiyor
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
