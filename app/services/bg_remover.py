from transformers import AutoModelForImageSegmentation
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
import numpy as np
import io
import base64
import time
from app.models.schemas import BackgroundRemovalResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "briaai/RMBG-1.4"
model = None

transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    global model
    if model is None:
        print(f"[BgRemover] Loading {MODEL_NAME}...")
        model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        model.eval()
        print("[BgRemover] Model loaded.")


def remove_background(image: Image.Image) -> tuple[BackgroundRemovalResponse, float]:
    load_model()

    t0 = time.time()
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    if isinstance(outputs, (list, tuple)):
        mask = outputs[0]
    elif hasattr(outputs, "pred_masks"):
        mask = outputs.pred_masks
    else:
        mask = outputs

    if isinstance(mask, (list, tuple)):
        mask = mask[0]
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]

    mask = torch.sigmoid(mask)

    # Threshold ile keskin maske
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = torch.where(mask > 0.5, mask * 1.5, mask * 0.2)
    mask = torch.clamp(mask, 0, 1)

    mask_np = (mask.numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")
    mask_pil = mask_pil.resize(original_size, Image.LANCZOS)

    # Kenarları hafif yumuşat
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))

    result = image.convert("RGBA")
    result.putalpha(mask_pil)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    processing_ms = round((time.time() - t0) * 1000, 2)

    log_inference(
        endpoint="/analyze/remove-background",
        model_name=MODEL_NAME,
        processing_time_ms=processing_ms,
    )

    return BackgroundRemovalResponse(
        image_base64=f"data:image/png;base64,{b64}"
    ), processing_ms
