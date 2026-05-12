from pydantic import BaseModel
from typing import List, Optional


class ClassificationPrediction(BaseModel):
    label: str
    confidence: float


class ClassificationResponse(BaseModel):
    top_prediction: str
    confidence: float
    top_3: List[ClassificationPrediction]


class DescriptionResponse(BaseModel):
    raw: str
    cleaned: str


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    detected_objects: List[DetectedObject]
    total_objects: int


class BackgroundRemovalResponse(BaseModel):
    image_base64: str


class ProcessingTime(BaseModel):
    classification_ms: float
    description_ms: float
    detection_ms: float
    background_removal_ms: float
    total_ms: float


class UnifiedAnalysisResponse(BaseModel):
    classification: ClassificationResponse
    description: DescriptionResponse
    detection: DetectionResponse
    background_removed: str
    processing_time_ms: ProcessingTime


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
