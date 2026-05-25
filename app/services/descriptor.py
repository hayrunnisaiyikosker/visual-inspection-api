from PIL import Image
import time
from app.models.schemas import DescriptionResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "Salesforce/blip-image-captioning-base"

def describe_image(image: Image.Image) -> tuple[DescriptionResponse, float]:
    t0 = time.time()
    text = "a product image"
    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/describe", model_name=MODEL_NAME, processing_time_ms=processing_ms)
    return DescriptionResponse(raw=text, cleaned=text.capitalize()), processing_ms
