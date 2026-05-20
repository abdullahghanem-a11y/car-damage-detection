from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


# ── Detection Box ──────────────────────────────────────────────────────────
class DetectionBox(BaseModel):
    class_id:       int
    class_name:     str
    confidence:     float
    bbox:           List[float]
    severity_label: str
    severity_score: float
    severity_color: str


# ── Per-Image Result (YOLO output) ─────────────────────────────────────────
class DetectionResult(BaseModel):
    filename:         str
    verified:         bool
    vehicle_type:     Optional[str]
    rejection_reason: Optional[str]
    total_instances:  int
    detections:       List[DetectionBox]
    annotated_image:  str
    overall_severity: Optional[dict]


# ── Aggregated Damage (multi-angle) ────────────────────────────────────────
class AggregatedDamage(BaseModel):
    class_name:     str
    count:          int
    severity_label: str
    severity_score: float
    severity_color: str
    max_confidence: float


# ── Car Group (multi-angle) ────────────────────────────────────────────────
class CarGroup(BaseModel):
    vehicle_type:      str
    image_count:       int
    images:            List[DetectionResult]
    aggregated_damage: List[AggregatedDamage]
    total_instances:   int
    overall_severity:  dict


# ── Rejected Image ─────────────────────────────────────────────────────────
class RejectedImage(BaseModel):
    filename: str
    reason:   str


# ── Batch Detection Response (from YOLO pipeline) ──────────────────────────
class BatchDetectionResponse(BaseModel):
    session_id:      int
    session_name:    str
    total_images:    int
    verified_images: int
    rejected_images: List[RejectedImage]
    car_groups:      List[CarGroup]
    total_instances: int


# ── Single Detection DB Response ───────────────────────────────────────────
class DetectionResponse(BaseModel):
    id:               int
    session_id:       int
    image_filename:   str
    image_path:       str
    annotated_image:  Optional[str]
    results:          List[dict]
    total_instances:  int
    model_version:    str
    verified:         int
    vehicle_type:     Optional[str]
    overall_severity: Optional[str]
    created_at:       datetime

    class Config:
        from_attributes = True


# ── Session Schemas ────────────────────────────────────────────────────────
class SessionResponse(BaseModel):
    id:               int
    name:             str
    model_version:    str
    total_instances:  int
    overall_severity: Optional[str]
    created_at:       datetime
    updated_at:       datetime
    detections:       List[DetectionResponse] = []

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    id:               int
    name:             str
    model_version:    str
    total_instances:  int
    overall_severity: Optional[str]
    created_at:       datetime
    updated_at:       datetime
    image_count:      int           # number of detections in session

    class Config:
        from_attributes = True


# ── Rename Request ─────────────────────────────────────────────────────────
class RenameRequest(BaseModel):
    name: str


# ── Delete Images Request ──────────────────────────────────────────────────
class DeleteImagesRequest(BaseModel):
    detection_ids: List[int]       # list of detection IDs to delete