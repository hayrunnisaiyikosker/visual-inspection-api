from PIL import Image
import time
from app.models.schemas import DetectionResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "facebook/detr-resnet-50"

def detect_objects(image: Image.Image) -> tuple[DetectionResponse, float]:
    t0 = time.time()
    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/detect", model_name=MODEL_NAME, processing_time_ms=processing_ms, extra_metrics={"total_objects": 0})
    return DetectionResponse(detected_objects=[], total_objects=0), processing_ms
