from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import time
from app.models.schemas import DetectionResponse, DetectedObject, BoundingBox
from app.utils.mlflow_logger import log_inference

MODEL_NAME = "facebook/detr-resnet-50"
processor = None
model = None


def load_model():
    global processor, model
    if model is None:
        print(f"[Detector] Loading {MODEL_NAME}...")
        processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
        model = DetrForObjectDetection.from_pretrained(MODEL_NAME)
        model.eval()
        print("[Detector] Model loaded.")


def detect_objects(image: Image.Image) -> tuple[DetectionResponse, float]:
    load_model()

    t0 = time.time()
    width, height = image.size
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )[0]

    detected = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x_min, y_min, x_max, y_max = box.tolist()
        detected.append(
            DetectedObject(
                label=model.config.id2label[label_id.item()],
                confidence=round(score.item(), 4),
                bbox=BoundingBox(
                    x_min=round(x_min / width, 4),
                    y_min=round(y_min / height, 4),
                    x_max=round(x_max / width, 4),
                    y_max=round(y_max / height, 4),
                ),
            )
        )

    processing_ms = round((time.time() - t0) * 1000, 2)

    log_inference(
        endpoint="/analyze/detect",
        model_name=MODEL_NAME,
        processing_time_ms=processing_ms,
        extra_metrics={"total_objects": len(detected)},
    )

    return DetectionResponse(detected_objects=detected, total_objects=len(detected)), processing_ms
