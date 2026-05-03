from pydantic import BaseModel
from datetime import datetime
from typing import List, Any


class DetectionBox(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[float]  # [x1, y1, x2, y2]


class DetectionResult(BaseModel):
    filename:        str
    total_instances: int
    detections:      List[DetectionBox]
    annotated_image: str  # base64 encoded image


class DetectionResponse(BaseModel):
    id:              int
    image_filename:  str
    total_instances: int
    results:         List[DetectionBox]
    model_version:   str
    created_at:      datetime

    class Config:
        from_attributes = True


class BatchDetectionResponse(BaseModel):
    total_images:    int
    total_instances: int
    results:         List[DetectionResult]