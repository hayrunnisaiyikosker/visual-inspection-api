from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from typing import List
from app.models.schemas import ClassificationPrediction, ClassificationResponse


MODEL_NAME = "google/vit-base-patch16-224"

processor = None
model = None


def load_model():
    global processor, model
    if model is None:
        print(f"[Classifier] Loading {MODEL_NAME}...")
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("[Classifier] Model loaded.")


def classify_image(image: Image.Image) -> ClassificationResponse:
    load_model()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    top3_indices = torch.topk(probs, k=3).indices.tolist()

    top3 = []
    for idx in top3_indices:
        label = model.config.id2label[idx]
        confidence = round(probs[idx].item(), 4)
        top3.append(ClassificationPrediction(label=label, confidence=confidence))

    return ClassificationResponse(
        top_prediction=top3[0].label,
        confidence=top3[0].confidence,
        top_3=top3,
    )
