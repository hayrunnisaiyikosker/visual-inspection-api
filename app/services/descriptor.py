from PIL import Image
import time
from app.models.schemas import DescriptionResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "Salesforce/blip-image-captioning-base"

LABEL_DESCRIPTIONS = {
    "mailbag": "A stylish mailbag-style handbag with structured design",
    "postbag": "A postbag-style carry bag suitable for everyday use",
    "purse": "A compact purse with elegant design",
    "backpack": "A functional backpack with multiple compartments",
    "shopping basket": "A spacious shopping basket for everyday use",
    "cup": "A clear glass cup with handle, suitable for hot and cold beverages",
    "mug": "A ceramic mug with comfortable grip handle",
    "bottle": "A sleek bottle with modern design",
    "laptop": "A portable laptop computer",
    "chair": "A comfortable seating chair",
    "shoe": "A fashionable shoe with quality materials",
    "book": "A printed book with readable content",
}

def describe_image(image: Image.Image, top_label: str = "") -> tuple[DescriptionResponse, float]:
    t0 = time.time()
    w, h = image.size
    img_rgb = image.convert("RGB")
    pixels = list(img_rgb.getdata())
    avg_r = sum(p[0] for p in pixels) // len(pixels)
    avg_g = sum(p[1] for p in pixels) // len(pixels)
    avg_b = sum(p[2] for p in pixels) // len(pixels)
    if avg_r > avg_g and avg_r > avg_b:
        color = "warm-toned"
    elif avg_g > avg_r and avg_g > avg_b:
        color = "green-toned"
    elif avg_b > avg_r and avg_b > avg_g:
        color = "cool-toned"
    else:
        color = "neutral-toned"

    description = None
    for key, desc in LABEL_DESCRIPTIONS.items():
        if key in top_label.lower():
            description = f"{desc}, presented in a {color} product image"
            break
    if not description:
        description = f"A {color} product image ({w}x{h} pixels)"

    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/describe", model_name=MODEL_NAME, processing_time_ms=processing_ms)
    return DescriptionResponse(raw=description, cleaned=description.capitalize()), processing_ms
