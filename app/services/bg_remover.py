from PIL import Image
import io
import base64
import time
from app.models.schemas import BackgroundRemovalResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "briaai/RMBG-1.4"

def remove_background(image: Image.Image) -> tuple[BackgroundRemovalResponse, float]:
    t0 = time.time()
    try:
        from rembg import remove
        result = remove(image)
    except Exception as e:
        print(f"[BgRemover] rembg failed: {e}, returning original")
        result = image.convert("RGBA")
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/remove-background", model_name=MODEL_NAME, processing_time_ms=processing_ms)
    return BackgroundRemovalResponse(image_base64=f"data:image/png;base64,{b64}"), processing_ms
