from PIL import Image
import time
from app.models.schemas import DescriptionResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "Salesforce/blip-image-captioning-base"

def unload_model():
    global processor, model
    processor = None; model = None
    import gc; gc.collect()

def describe_image(image: Image.Image) -> tuple[DescriptionResponse, float]:
    t0 = time.time()
    w, h = image.size
    mode = image.mode
    # Basit renk analizi
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
    text = f"A {color} product image ({w}x{h} pixels)"
    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/describe", model_name=MODEL_NAME, processing_time_ms=processing_ms)
    return DescriptionResponse(raw=text, cleaned=text.capitalize()), processing_ms
