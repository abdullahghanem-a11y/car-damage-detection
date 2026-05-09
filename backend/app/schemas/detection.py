from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


# ── Single Detection Box ───────────────────────
class DetectionBox(BaseModel):
    class_id:       int
    class_name:     str
    confidence:     float
    bbox:           List[float]
    severity_label: str
    severity_score: float
    severity_color: str


# ── Per-Image Result ───────────────────────────
class DetectionResult(BaseModel):
    filename:         str
    verified:         bool
    vehicle_type:     Optional[str]
    rejection_reason: Optional[str]
    total_instances:  int
    detections:       List[DetectionBox]
    annotated_image:  str
    overall_severity: Optional[dict]


# ── Aggregated Damage (multi-angle) ───────────
class AggregatedDamage(BaseModel):
    class_name:     str
    count:          int
    severity_label: str
    severity_score: float
    severity_color: str
    max_confidence: float


# ── Car Group (multi-angle) ───────────────────
class CarGroup(BaseModel):
    vehicle_type:      str
    image_count:       int
    images:            List[DetectionResult]
    aggregated_damage: List[AggregatedDamage]
    total_instances:   int
    overall_severity:  dict


# ── Rejected Image ────────────────────────────
class RejectedImage(BaseModel):
    filename: str
    reason:   str


# ── Batch Response ────────────────────────────
class BatchDetectionResponse(BaseModel):
    total_images:    int
    verified_images: int
    rejected_images: List[RejectedImage]
    car_groups:      List[CarGroup]
    total_instances: int


# ── History Response ──────────────────────────
class DetectionResponse(BaseModel):
    id:              int
    image_filename:  str
    total_instances: int
    results:         List[dict]
    model_version:   str
    created_at:      datetime

    class Config:
        from_attributes = True