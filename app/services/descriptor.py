from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time
from app.models.schemas import DescriptionResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = None
model = None


def load_model():
    global processor, model
    if model is None:
        print(f"[Descriptor] Loading {MODEL_NAME}...")
        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.eval()
        print("[Descriptor] Model loaded.")


def describe_image(image: Image.Image) -> tuple[DescriptionResponse, float]:
    load_model()

    t0 = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    raw_text = processor.decode(output[0], skip_special_tokens=True)
    cleaned_text = raw_text.strip().capitalize()
    processing_ms = round((time.time() - t0) * 1000, 2)

    log_inference(
        endpoint="/analyze/describe",
        model_name=MODEL_NAME,
        processing_time_ms=processing_ms,
    )

    return DescriptionResponse(raw=raw_text, cleaned=cleaned_text), processing_ms
