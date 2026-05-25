from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import gc
import time
from app.models.schemas import ClassificationPrediction, ClassificationResponse
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "google/vit-base-patch16-224"

def classify_image(image: Image.Image) -> tuple[ClassificationResponse, float]:
    print(f"[Classifier] Loading {MODEL_NAME}...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    t0 = time.time()
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
    processing_ms = round((time.time() - t0) * 1000, 2)
    log_inference(endpoint="/analyze/classify", model_name=MODEL_NAME, processing_time_ms=processing_ms, extra_metrics={"confidence": top3[0].confidence})
    result = ClassificationResponse(top_prediction=top3[0].label, confidence=top3[0].confidence, top_3=top3)
    del model, processor
    gc.collect()
    return result, processing_ms
