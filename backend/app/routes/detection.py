from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession
from typing import List
from dotenv import load_dotenv
from pathlib import Path
import os

from app.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_active_user
from app.models.detection import Detection, Session
from app.schemas.detection import BatchDetectionResponse
from app.services.yolo_service import run_batch_inference

load_dotenv()

ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png").split(",")
MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", 10))
MODEL_VERSION      = os.getenv("MODEL_VERSION", "v2.0/best.pt")
UPLOADS_DIR        = Path(os.getenv("UPLOADS_DIR", "uploads"))

router = APIRouter(prefix="/api/detect", tags=["Detection"])


def validate_file(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )


def save_raw_image(session_id: int, filename: str, image_bytes: bytes) -> str:
    """
    Save raw image bytes to uploads/{session_id}/{filename}.
    Returns the relative path string.
    """
    session_dir = UPLOADS_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename — remove path separators
    safe_name = Path(filename).name
    file_path = session_dir / safe_name
    file_path.write_bytes(image_bytes)

    return str(file_path)


@router.post("/", response_model=BatchDetectionResponse)
async def detect_damage(
    files:        List[UploadFile] = File(..., media_type="image/*"),
    session_name: str              = Query(..., description="Name for this detection session"),
    db:           DBSession        = Depends(get_db),
    current_user: User             = Depends(get_current_active_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per request")
    if not session_name or not session_name.strip():
        raise HTTPException(status_code=400, detail="session_name is required")

    # ── Read + validate ────────────────────────────────────────────────────
    images_data = []
    for file in files:
        validate_file(file)
        image_bytes = await file.read()
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit")
        images_data.append({"filename": file.filename, "bytes": image_bytes})

    # ── Run YOLO ───────────────────────────────────────────────────────────
    batch_result = run_batch_inference(images_data)

    # ── Session severity ───────────────────────────────────────────────────
    all_severities = [
        group.get("overall_severity", {}).get("label")
        for group in batch_result["car_groups"]
        if group.get("overall_severity", {}).get("label") not in (None, "None")
    ]
    severity_rank    = {"Severe": 3, "Moderate": 2, "Minor": 1}
    session_severity = max(all_severities, key=lambda s: severity_rank.get(s, 0)) if all_severities else None

    # ── Create session ─────────────────────────────────────────────────────
    session = Session(
        user_id          = current_user.id,
        name             = session_name.strip(),
        model_version    = MODEL_VERSION,
        total_instances  = batch_result["total_instances"],
        overall_severity = session_severity,
    )
    db.add(session)
    db.flush()  # get session.id

    img_bytes_map = {d["filename"]: d["bytes"] for d in images_data}

    # ── Save verified detections ───────────────────────────────────────────
    for group in batch_result["car_groups"]:
        for img_result in group["images"]:
            raw_bytes  = img_bytes_map.get(img_result["filename"], b"")
            # Write raw image to disk
            image_path = save_raw_image(session.id, img_result["filename"], raw_bytes) if raw_bytes else ""

            detection = Detection(
                session_id       = session.id,
                image_filename   = img_result["filename"],
                image_path       = image_path,
                annotated_image  = img_result.get("annotated_image"),
                results          = img_result["detections"],
                total_instances  = img_result["total_instances"],
                model_version    = MODEL_VERSION,
                verified         = 1 if img_result["verified"] else 0,
                vehicle_type     = img_result.get("vehicle_type"),
                overall_severity = img_result.get("overall_severity", {}).get("label"),
            )
            db.add(detection)

    # ── Save rejected images ───────────────────────────────────────────────
    for rejected in batch_result["rejected_images"]:
        raw_bytes  = img_bytes_map.get(rejected["filename"], b"")
        image_path = save_raw_image(session.id, rejected["filename"], raw_bytes) if raw_bytes else ""

        detection = Detection(
            session_id       = session.id,
            image_filename   = rejected["filename"],
            image_path       = image_path,
            annotated_image  = None,
            results          = [],
            total_instances  = 0,
            model_version    = MODEL_VERSION,
            verified         = 0,
            vehicle_type     = None,
            overall_severity = "Rejected",
        )
        db.add(detection)

    db.commit()
    db.refresh(session)

    batch_result["session_id"]   = session.id
    batch_result["session_name"] = session.name

    return batch_result