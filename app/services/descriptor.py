from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from app.models.schemas import DescriptionResponse


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


def describe_image(image: Image.Image) -> DescriptionResponse:
    load_model()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    raw_text = processor.decode(output[0], skip_special_tokens=True)
    cleaned_text = raw_text.strip().capitalize()

    return DescriptionResponse(raw=raw_text, cleaned=cleaned_text)
