from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from dotenv import load_dotenv
import os

from app.database import get_db
from app.models.detection import Detection
from app.schemas.detection import BatchDetectionResponse, DetectionResponse
from app.services.yolo_service import run_batch_inference

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
    - Verifies each image contains a car
    - Groups images of the same car together
    - Returns severity scores per detection
    - Aggregates multi-angle results per car
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per request")

    # Read all files
    images_data = []
    for file in files:
        validate_file(file)
        image_bytes = await file.read()

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code = 400,
                detail      = f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit"
            )

        images_data.append({
            "filename": file.filename,
            "bytes":    image_bytes
        })

    # Run full pipeline
    batch_result = run_batch_inference(images_data)

    # Save each verified detection to database
    for group in batch_result["car_groups"]:
        for img_result in group["images"]:
            detection = Detection(
                image_filename   = img_result["filename"],
                image_path       = f"uploads/{img_result['filename']}",
                results          = img_result["detections"],
                total_instances  = img_result["total_instances"],
                model_version    = MODEL_VERSION,
                verified         = 1 if img_result["verified"] else 0,
                vehicle_type     = img_result.get("vehicle_type"),
                overall_severity = img_result.get("overall_severity", {}).get("label")
            )
            db.add(detection)

    # Save rejected images to database too
    for rejected in batch_result["rejected_images"]:
        detection = Detection(
            image_filename   = rejected["filename"],
            image_path       = f"uploads/{rejected['filename']}",
            results          = [],
            total_instances  = 0,
            model_version    = MODEL_VERSION,
            verified         = 0,
            vehicle_type     = None,
            overall_severity = "Rejected"
        )
        db.add(detection)

    db.commit()

    return batch_result


@router.get("/history", response_model=List[DetectionResponse])
def get_history(
    skip:  int     = 0,
    limit: int     = 20,
    db:    Session = Depends(get_db)
):
    """Get detection history with pagination."""
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