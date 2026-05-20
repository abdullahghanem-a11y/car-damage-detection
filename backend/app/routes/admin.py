"""
routes_admin.py
---------------
Admin-only endpoints. All require require_admin dependency.

GET  /api/admin/stats          — dashboard stats
GET  /api/admin/health         — app + DB + model health
GET  /api/admin/users          — list all users
PATCH /api/admin/users/{id}    — enable/disable user
GET  /api/admin/logs           — activity logs with filter/pagination
GET  /api/admin/model          — current model info + metrics
PATCH /api/admin/model         — switch active model version
GET  /api/admin/feedback       — all user feedback
"""

import os
import csv
import json
import time
import psutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, desc
from sqlalchemy.orm import Session as DBSession

from app.database import get_db, engine
from app.models.user      import User, UserRole
from app.models.detection import Session as DetSession, Detection
from app.models.feedback  import Feedback
from app.models.log       import ActivityLog
from app.schemas.auth     import UserResponse, FeedbackResponse
from app.services.auth_service import require_admin

APP_START_TIME = time.time()

router = APIRouter(prefix="/api/admin", tags=["Admin"])

# ── Model config (in-memory, updated by switcher) ──────────────────────────
_active_model_version = os.getenv("MODEL_VERSION", "v2.0/best.pt")


# ── GET /api/admin/stats ───────────────────────────────────────────────────
@router.get("/stats")
def get_stats(db: DBSession = Depends(get_db), _=Depends(require_admin)):
    """Dashboard overview stats."""

    total_users    = db.query(func.count(User.id)).scalar()
    total_sessions = db.query(func.count(DetSession.id)).scalar()
    total_detections = db.query(func.count(Detection.id)).scalar()
    total_instances  = db.query(func.sum(DetSession.total_instances)).scalar() or 0

    # Severity distribution
    severity_dist = db.query(
        DetSession.overall_severity,
        func.count(DetSession.id)
    ).group_by(DetSession.overall_severity).all()

    # Damage type distribution (from detection results JSON)
    detections_all = db.query(Detection.results).filter(
        Detection.verified == 1,
        Detection.results != None,
    ).all()

    damage_counts = {}
    for (results,) in detections_all:
        if isinstance(results, list):
            for det in results:
                cls = det.get("class_name", "unknown")
                damage_counts[cls] = damage_counts.get(cls, 0) + 1

    # Sessions per day (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    daily_sessions = db.query(
        func.date(DetSession.created_at).label("date"),
        func.count(DetSession.id).label("count")
    ).filter(
        DetSession.created_at >= thirty_days_ago
    ).group_by(func.date(DetSession.created_at))\
     .order_by(func.date(DetSession.created_at)).all()

    # New users per day (last 30 days)
    daily_users = db.query(
        func.date(User.created_at).label("date"),
        func.count(User.id).label("count")
    ).filter(
        User.created_at >= thirty_days_ago
    ).group_by(func.date(User.created_at))\
     .order_by(func.date(User.created_at)).all()

    # Recent sessions
    recent = db.query(DetSession)\
               .order_by(DetSession.created_at.desc())\
               .limit(5).all()

    return {
        "overview": {
            "total_users":      total_users,
            "total_sessions":   total_sessions,
            "total_detections": total_detections,
            "total_instances":  total_instances,
        },
        "severity_distribution": [
            {"severity": s or "Unknown", "count": c}
            for s, c in severity_dist
        ],
        "damage_distribution": [
            {"class_name": k, "count": v}
            for k, v in sorted(damage_counts.items(), key=lambda x: -x[1])
        ],
        "daily_sessions": [
            {"date": str(d), "count": c} for d, c in daily_sessions
        ],
        "daily_users": [
            {"date": str(d), "count": c} for d, c in daily_users
        ],
        "recent_sessions": [
            {
                "id": s.id, "name": s.name,
                "total_instances": s.total_instances,
                "overall_severity": s.overall_severity,
                "created_at": s.created_at.isoformat(),
            }
            for s in recent
        ],
    }


# ── GET /api/admin/health ──────────────────────────────────────────────────
@router.get("/health")
def get_health(db: DBSession = Depends(get_db), _=Depends(require_admin)):
    """App, DB, and model health check."""

    # DB check
    db_ok = False
    try:
        db.execute(engine.dialect.statement_compiler(engine.dialect, None).__class__.__mro__[0].__new__(engine.dialect.statement_compiler))
    except Exception:
        pass
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # Model check
    from app.services.yolo_service import load_damage_model
    model_ok = False
    try:
        load_damage_model()
        model_ok = True
    except Exception:
        model_ok = False

    # System stats
    process = psutil.Process(os.getpid())
    mem_mb  = round(process.memory_info().rss / 1024 / 1024, 1)
    cpu_pct = psutil.cpu_percent(interval=0.1)
    uptime_s = int(time.time() - APP_START_TIME)
    uptime_str = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m {uptime_s % 60}s"

    return {
        "status":       "healthy" if db_ok and model_ok else "degraded",
        "database":     {"status": "connected" if db_ok else "error"},
        "model":        {"status": "loaded" if model_ok else "error", "version": _active_model_version},
        "system": {
            "uptime":     uptime_str,
            "memory_mb":  mem_mb,
            "cpu_pct":    cpu_pct,
        },
    }


# ── GET /api/admin/users ───────────────────────────────────────────────────
@router.get("/users")
def list_users(
    skip:   int = Query(0,  ge=0),
    limit:  int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    db:     DBSession     = Depends(get_db),
    _=Depends(require_admin),
):
    query = db.query(User)
    if search:
        query = query.filter(
            (User.username.ilike(f"%{search}%")) |
            (User.email.ilike(f"%{search}%"))
        )
    total = query.count()
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "users": [
            {
                "id":         u.id,
                "username":   u.username,
                "email":      u.email,
                "role":       u.role,
                "is_active":  u.is_active,
                "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None,
                "sessions":   db.query(func.count(DetSession.id)).filter(DetSession.user_id == u.id).scalar(),
            }
            for u in users
        ]
    }


# ── PATCH /api/admin/users/{id} ───────────────────────────────────────────
@router.patch("/users/{user_id}")
def toggle_user(
    user_id: int,
    body:    dict,
    db:      DBSession = Depends(get_db),
    admin:   User      = Depends(require_admin),
):
    """Enable or disable a user account."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role == UserRole.admin:
        raise HTTPException(status_code=400, detail="Cannot modify admin accounts")

    user.is_active = body.get("is_active", user.is_active)
    db.commit()
    return {"message": f"User {'enabled' if user.is_active else 'disabled'}", "is_active": user.is_active}


# ── GET /api/admin/logs ────────────────────────────────────────────────────
@router.get("/logs")
def get_logs(
    skip:   int           = Query(0,  ge=0),
    limit:  int           = Query(50, ge=1, le=200),
    action: Optional[str] = Query(None),
    user_id:Optional[int] = Query(None),
    status: Optional[int] = Query(None),
    db:     DBSession     = Depends(get_db),
    _=Depends(require_admin),
):
    query = db.query(ActivityLog)
    if action:   query = query.filter(ActivityLog.action == action)
    if user_id:  query = query.filter(ActivityLog.user_id == user_id)
    if status:   query = query.filter(ActivityLog.status == status)

    total = query.count()
    logs  = query.order_by(ActivityLog.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "logs": [
            {
                "id":         l.id,
                "user_id":    l.user_id,
                "username":   l.user.username if l.user else "anonymous",
                "action":     l.action,
                "method":     l.method,
                "path":       l.path,
                "status":     l.status,
                "ip_address": l.ip_address,
                "details":    l.details,
                "created_at": l.created_at.isoformat(),
            }
            for l in logs
        ]
    }


# ── GET /api/admin/model ───────────────────────────────────────────────────
@router.get("/model")
def get_model_info(_=Depends(require_admin)):
    """Return current model version, available versions, and training metrics."""
    global _active_model_version

    metrics = _load_metrics()
    available = _get_available_versions()

    return {
        "active_version": _active_model_version,
        "available_versions": available,
        "metrics": metrics,
    }


# ── PATCH /api/admin/model ────────────────────────────────────────────────
@router.patch("/model")
def switch_model(body: dict, _=Depends(require_admin)):
    """Switch active model version and reload in memory immediately."""
    global _active_model_version
    from app.services.yolo_service import switch_model_version

    new_version = body.get("version", "").strip()
    if not new_version:
        raise HTTPException(status_code=400, detail="version is required")

    available = _get_available_versions()
    if new_version not in available:
        raise HTTPException(status_code=400, detail=f"Version '{new_version}' not found. Available: {available}")

    try:
        switch_model_version(new_version)
        _active_model_version = new_version
        return {"message": f"Model switched to {new_version}", "active_version": new_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


# ── GET /api/admin/feedback ───────────────────────────────────────────────
@router.get("/feedback")
def get_feedback(
    skip:  int = Query(0,  ge=0),
    limit: int = Query(50, ge=1, le=200),
    db:    DBSession = Depends(get_db),
    _=Depends(require_admin),
):
    total    = db.query(func.count(Feedback.id)).scalar()
    feedbacks = db.query(Feedback).order_by(Feedback.created_at.desc()).offset(skip).limit(limit).all()
    avg_rating = db.query(func.avg(Feedback.rating)).filter(Feedback.rating != None).scalar()

    return {
        "total":      total,
        "avg_rating": round(float(avg_rating), 2) if avg_rating else None,
        "feedbacks": [
            {
                "id":         f.id,
                "username":   f.user.username if f.user else "deleted",
                "message":    f.message,
                "rating":     f.rating,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedbacks
        ]
    }


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_available_versions() -> list[str]:
    """Scan weights directory for available model versions."""
    weights_dir = Path(os.getenv("MODEL_CACHE_DIR", "./weights"))
    versions    = []
    for pt_file in weights_dir.rglob("best.pt"):
        # e.g. weights/models--abdullahg7--cardd-yolov8s/snapshots/.../v2.0/best.pt
        parts = pt_file.parts
        for i, part in enumerate(parts):
            if part.startswith("v") and i + 1 < len(parts) and parts[i+1] == "best.pt":
                versions.append(f"{part}/best.pt")
    return sorted(set(versions)) or ["v2.0/best.pt"]


def _load_metrics() -> dict:
    """Load training metrics from results CSV files."""
    results = {}

    csv_map = {
        "v1.0/best.pt": Path(__file__).parent.parent / "data" / "results_v1.csv",
        "v2.0/best.pt": Path(__file__).parent.parent / "data" / "results_v2.csv",
    }

    for version, csv_path in csv_map.items():
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})

        if not rows:
            continue

        # Best epoch = highest mAP50(B)
        best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
        last = rows[-1]

        results[version] = {
            "epochs":      len(rows),
            "best_epoch":  int(float(best.get("epoch", 0))),
            "best": {
                "mAP50_box":    round(float(best.get("metrics/mAP50(B)",    0)) * 100, 2),
                "mAP50_mask":   round(float(best.get("metrics/mAP50(M)",    0) or 0) * 100, 2),
                "mAP50_95_box": round(float(best.get("metrics/mAP50-95(B)", 0)) * 100, 2),
                "precision":    round(float(best.get("metrics/precision(B)", 0)) * 100, 2),
                "recall":       round(float(best.get("metrics/recall(B)",   0)) * 100, 2),
            },
            "final": {
                "mAP50_box":    round(float(last.get("metrics/mAP50(B)",    0)) * 100, 2),
                "mAP50_mask":   round(float(last.get("metrics/mAP50(M)",    0) or 0) * 100, 2),
                "precision":    round(float(last.get("metrics/precision(B)", 0)) * 100, 2),
                "recall":       round(float(last.get("metrics/recall(B)",   0)) * 100, 2),
            },
            # Full training curve for charts
            "curve": [
                {
                    "epoch":     int(float(r.get("epoch", 0))),
                    "mAP50_box": round(float(r.get("metrics/mAP50(B)", 0)) * 100, 2),
                    "mAP50_mask":round(float(r.get("metrics/mAP50(M)", 0) or 0) * 100, 2),
                    "precision": round(float(r.get("metrics/precision(B)", 0)) * 100, 2),
                    "recall":    round(float(r.get("metrics/recall(B)",    0)) * 100, 2),
                    "train_loss":round(float(r.get("train/box_loss", 0)), 4),
                    "val_loss":  round(float(r.get("val/box_loss",   0)), 4),
                }
                for r in rows
            ]
        }

    return results