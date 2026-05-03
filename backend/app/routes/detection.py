from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from dotenv import load_dotenv
import os

from app.database import get_db
from app.models.detection import Detection
from app.schemas.detection import BatchDetectionResponse, DetectionResponse
from app.services.yolo_service import run_inference

# ── Load .env ─────────────────────────────────
load_dotenv()

ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png").split(",")
MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", 10))
MODEL_VERSION      = os.getenv("MODEL_VERSION", "v1.0/best.pt")

router = APIRouter(prefix="/api/detect", tags=["Detection"])


def validate_file(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code = 400,
            detail      = f"File type .{ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )


@router.post("/", response_model=BatchDetectionResponse)
async def detect_damage(
    files: List[UploadFile] = File(..., media_type="image/*"),
    db:    Session          = Depends(get_db)
):
    """
    Detect car damage in one or more uploaded images.
    Returns annotated images with bounding boxes and detection details.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per request")

    all_results     = []
    total_instances = 0

    for file in files:
        validate_file(file)

        image_bytes = await file.read()

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code = 400,
                detail      = f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit"
            )

        result = run_inference(image_bytes, file.filename)

        detection = Detection(
            image_filename  = file.filename,
            image_path      = f"uploads/{file.filename}",
            results         = result["detections"],
            total_instances = result["total_instances"],
            model_version   = MODEL_VERSION
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)

        all_results.append(result)
        total_instances += result["total_instances"]

    return BatchDetectionResponse(
        total_images    = len(files),
        total_instances = total_instances,
        results         = all_results
    )


@router.get("/history", response_model=List[DetectionResponse])
def get_history(
    skip:  int     = 0,
    limit: int     = 20,
    db:    Session = Depends(get_db)
):
    """Get history of all past detections."""
    detections = db.query(Detection)\
                   .order_by(Detection.created_at.desc())\
                   .offset(skip)\
                   .limit(limit)\
                   .all()
    return detections


@router.get("/history/{detection_id}", response_model=DetectionResponse)
def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get a single detection by ID."""
    detection = db.query(Detection)\
                  .filter(Detection.id == detection_id)\
                  .first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection