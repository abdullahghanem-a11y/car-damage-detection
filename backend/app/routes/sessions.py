from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, case
from pathlib import Path
from typing import List, Optional
import os

from app.database import get_db
from app.models.detection import Detection, Session
from app.models.user import User, UserRole
from app.services.auth_service import get_current_active_user
from app.schemas.detection import (
    SessionResponse, SessionListResponse,
    RenameRequest, DeleteImagesRequest,
)

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


# ── Helpers ────────────────────────────────────────────────────────────────
def read_raw_image(image_path: str) -> bytes:
    """Read raw image bytes from disk. Raises 404 if not found."""
    path = Path(image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Raw image file not found: {image_path}")
    return path.read_bytes()


def save_raw_image(session_id: int, filename: str, image_bytes: bytes) -> str:
    """Save raw image to uploads/{session_id}/{filename}, return path string."""
    session_dir = UPLOADS_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    file_path = session_dir / safe_name
    file_path.write_bytes(image_bytes)
    return str(file_path)


# ── GET /api/sessions/ ─────────────────────────────────────────────────────
@router.get("/", response_model=List[SessionListResponse])
def list_sessions(
    skip:     int           = Query(0,  ge=0),
    limit:    int           = Query(20, ge=1, le=100),
    search:   Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    sort:     Optional[str] = Query("date_desc"),
    db:       DBSession     = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Admins see all sessions, regular users see only their own
    query = db.query(Session) if current_user.role == UserRole.admin else db.query(Session).filter(Session.user_id == current_user.id)

    if search:
        query = query.filter(Session.name.ilike(f"%{search}%"))
    if severity:
        query = query.filter(Session.overall_severity == severity)

    severity_case = case(
        (Session.overall_severity == "Severe",   3),
        (Session.overall_severity == "Moderate", 2),
        (Session.overall_severity == "Minor",    1),
        else_=0,
    )

    SORT_MAP = {
        "date_desc":      Session.created_at.desc(),
        "date_asc":       Session.created_at.asc(),
        "name_asc":       Session.name.asc(),
        "name_desc":      Session.name.desc(),
        "instances_desc": Session.total_instances.desc(),
        "instances_asc":  Session.total_instances.asc(),
        "severity_desc":  severity_case.desc(),
        "severity_asc":   severity_case.asc(),
    }

    query = query.order_by(SORT_MAP.get(sort, Session.created_at.desc()))
    sessions = query.offset(skip).limit(limit).all()

    result = []
    for s in sessions:
        image_count = db.query(func.count(Detection.id))\
                        .filter(Detection.session_id == s.id).scalar()
        result.append(SessionListResponse(
            id=s.id, name=s.name, model_version=s.model_version,
            total_instances=s.total_instances, overall_severity=s.overall_severity,
            created_at=s.created_at, updated_at=s.updated_at, image_count=image_count,
        ))
    return result


# ── GET /api/sessions/{id} ─────────────────────────────────────────────────
@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: DBSession = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.role != UserRole.admin and session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return session


# ── GET /api/sessions/{id}/images/{det_id}/raw ────────────────────────────
@router.get("/{session_id}/images/{detection_id}/raw")
def get_raw_image(session_id: int, detection_id: int, db: DBSession = Depends(get_db)):
    """
    Serve the raw (unannotated) image file from disk.
    Returns image bytes with explicit CORS headers so react-easy-crop
    can load it cross-origin via canvas without taint issues.
    """
    detection = db.query(Detection).filter(
        Detection.id         == detection_id,
        Detection.session_id == session_id,
    ).first()

    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    if not detection.image_path or not Path(detection.image_path).exists():
        raise HTTPException(status_code=404, detail="Raw image file not found on disk")

    image_bytes = Path(detection.image_path).read_bytes()

    # Detect content type from extension
    ext = Path(detection.image_filename).suffix.lower()
    media_type = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    return Response(
        content    = image_bytes,
        media_type = media_type,
        headers    = {
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET",
            "Cache-Control":                "no-cache",
        },
    )


# ── PATCH /api/sessions/{id}/rename ───────────────────────────────────────
@router.patch("/{session_id}/rename", response_model=SessionListResponse)
def rename_session(session_id: int, body: RenameRequest, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Session name cannot be empty")

    session.name = name
    db.commit()
    db.refresh(session)

    image_count = db.query(func.count(Detection.id))\
                    .filter(Detection.session_id == session_id).scalar()

    return SessionListResponse(
        id=session.id, name=session.name, model_version=session.model_version,
        total_instances=session.total_instances, overall_severity=session.overall_severity,
        created_at=session.created_at, updated_at=session.updated_at, image_count=image_count,
    )


# ── DELETE /api/sessions/{id} ─────────────────────────────────────────────
@router.delete("/{session_id}")
def delete_session(session_id: int, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete image files from disk
    session_dir = UPLOADS_DIR / str(session_id)
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    db.delete(session)
    db.commit()
    return {"message": f"Session {session_id} deleted successfully"}


# ── DELETE /api/sessions/{id}/images ─────────────────────────────────────
@router.delete("/{session_id}/images")
def delete_session_images(
    session_id: int,
    body:       DeleteImagesRequest,
    db:         DBSession = Depends(get_db),
):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not body.detection_ids:
        raise HTTPException(status_code=400, detail="No detection IDs provided")

    detections = db.query(Detection).filter(
        Detection.id.in_(body.detection_ids),
        Detection.session_id == session_id,
    ).all()

    if not detections:
        raise HTTPException(status_code=404, detail="No matching detections found")

    for det in detections:
        # Delete raw image file from disk
        if det.image_path:
            p = Path(det.image_path)
            if p.exists():
                p.unlink()
        db.delete(det)

    db.flush()

    remaining = db.query(func.count(Detection.id))\
                  .filter(Detection.session_id == session_id).scalar()

    if remaining == 0:
        session_dir = UPLOADS_DIR / str(session_id)
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        db.delete(session)
        db.commit()
        return {"message": "All images deleted — session removed", "session_deleted": True}

    new_total = db.query(func.sum(Detection.total_instances))\
                  .filter(Detection.session_id == session_id).scalar() or 0
    session.total_instances = new_total
    db.commit()

    return {
        "message":          f"{len(detections)} image(s) deleted",
        "session_deleted":  False,
        "remaining_images": remaining,
    }


# ── PATCH /api/sessions/{id}/images/{det_id} ─────────────────────────────
@router.patch("/{session_id}/images/{detection_id}")
def update_detection_image(
    session_id:   int,
    detection_id: int,
    body:         dict,
    db:           DBSession = Depends(get_db),
):
    """
    Overwrite raw image on disk with edited version (base64).
    """
    import base64

    detection = db.query(Detection).filter(
        Detection.id         == detection_id,
        Detection.session_id == session_id,
    ).first()

    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    image_data = body.get("image_data")
    if not image_data:
        raise HTTPException(status_code=400, detail="image_data is required")

    # Write edited image to disk (overwrite)
    image_bytes = base64.b64decode(image_data)
    image_path  = save_raw_image(session_id, detection.image_filename, image_bytes)
    detection.image_path = image_path
    db.commit()

    return {"message": "Image updated on disk successfully"}


# ── POST /api/sessions/{id}/images/{det_id}/redetect ──────────────────────
@router.post("/{session_id}/images/{detection_id}/redetect")
def redetect_image(
    session_id:   int,
    detection_id: int,
    db:           DBSession = Depends(get_db),
):
    """Re-run YOLO on the raw image stored on disk."""
    from app.services.yolo_service import run_inference

    detection = db.query(Detection).filter(
        Detection.id         == detection_id,
        Detection.session_id == session_id,
    ).first()

    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    # Read raw image from disk
    image_bytes = read_raw_image(detection.image_path)

    result = run_inference(image_bytes, detection.image_filename)

    detection.results          = result.get("detections", [])
    detection.annotated_image  = result.get("annotated_image")
    detection.total_instances  = result.get("total_instances", 0)
    detection.overall_severity = result.get("overall_severity", {}).get("label") if result.get("overall_severity") else None
    detection.verified         = 1 if result.get("verified") else 0

    session = db.query(Session).filter(Session.id == session_id).first()
    if session:
        new_total = db.query(func.sum(Detection.total_instances))\
                      .filter(Detection.session_id == session_id).scalar() or 0
        session.total_instances = new_total

    db.commit()

    return {
        "message":          "Re-detection complete",
        "total_instances":  detection.total_instances,
        "overall_severity": detection.overall_severity,
        "verified":         detection.verified,
    }


# ── POST /api/sessions/{id}/images/{det_id}/copy ──────────────────────────
@router.post("/{session_id}/images/{detection_id}/copy")
def copy_detection_image(
    session_id:   int,
    detection_id: int,
    body:         dict,
    db:           DBSession = Depends(get_db),
):
    """
    Save edited image as NEW detection in same session.
    Body: { "image_data": "<base64>", "copy_name": "My copy name" }
    """
    import base64
    from app.services.yolo_service import run_inference

    original = db.query(Detection).filter(
        Detection.id         == detection_id,
        Detection.session_id == session_id,
    ).first()

    if not original:
        raise HTTPException(status_code=404, detail="Original detection not found")

    image_data = body.get("image_data")
    copy_name  = body.get("copy_name", "").strip()

    if not image_data:
        raise HTTPException(status_code=400, detail="image_data is required")
    if not copy_name:
        raise HTTPException(status_code=400, detail="copy_name is required")

    # Ensure it has an image extension
    if not any(copy_name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
        copy_name += ".jpg"

    # Write new image to disk
    image_bytes = base64.b64decode(image_data)
    image_path  = save_raw_image(session_id, copy_name, image_bytes)

    # Run YOLO on the copy
    result = run_inference(image_bytes, copy_name)

    new_detection = Detection(
        session_id       = session_id,
        image_filename   = copy_name,
        image_path       = image_path,
        annotated_image  = result.get("annotated_image"),
        results          = result.get("detections", []),
        total_instances  = result.get("total_instances", 0),
        model_version    = original.model_version,
        verified         = 1 if result.get("verified") else 0,
        vehicle_type     = result.get("vehicle_type"),
        overall_severity = result.get("overall_severity", {}).get("label") if result.get("overall_severity") else None,
    )
    db.add(new_detection)

    session = db.query(Session).filter(Session.id == session_id).first()
    if session:
        session.total_instances = (session.total_instances or 0) + new_detection.total_instances

    db.commit()
    db.refresh(new_detection)

    return {
        "message":          "Copy created and re-detected successfully",
        "detection_id":     new_detection.id,
        "total_instances":  new_detection.total_instances,
        "overall_severity": new_detection.overall_severity,
        "verified":         new_detection.verified,
    }