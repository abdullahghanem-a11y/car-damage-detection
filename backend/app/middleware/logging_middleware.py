"""
logging_middleware.py
---------------------
FastAPI middleware that logs every API request to the activity_logs table.
Placed at app/middleware/logging_middleware.py
"""

import json
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.log import ActivityLog
from app.services.auth_service import decode_access_token

# Paths to skip logging (health checks, static files)
SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class ActivityLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip non-API paths and health checks
        if request.url.path in SKIP_PATHS or not request.url.path.startswith("/api"):
            return await call_next(request)

        start    = time.time()
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 1)  # ms

        # Extract user_id from token if present
        user_id = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                payload = decode_access_token(auth_header[7:])
                user_id = int(payload.get("sub", 0)) or None
            except Exception:
                pass

        # Build action label from method + path
        path   = request.url.path
        method = request.method
        action = _build_action(method, path)

        # Build details
        details = json.dumps({
            "duration_ms": duration,
            "query":       str(request.url.query) or None,
        })

        ip = request.client.host if request.client else None

        # Write to DB (non-blocking best-effort)
        try:
            db: Session = SessionLocal()
            log = ActivityLog(
                user_id    = user_id,
                action     = action,
                method     = method,
                path       = path,
                details    = details,
                ip_address = ip,
                status     = response.status_code,
            )
            db.add(log)
            db.commit()
        except Exception:
            pass
        finally:
            db.close()

        return response


def _build_action(method: str, path: str) -> str:
    """Convert method + path to a human-readable action label."""
    p = path.strip("/").replace("/api/", "")
    parts = p.split("/")

    mapping = {
        ("POST",   "auth/login"):    "login",
        ("POST",   "auth/register"): "register",
        ("POST",   "auth/logout"):   "logout",
        ("POST",   "auth/refresh"):  "token_refresh",
        ("POST",   "detect"):        "detect",
        ("GET",    "sessions"):      "list_sessions",
        ("DELETE", "sessions"):      "delete_session",
        ("PATCH",  "sessions"):      "update_session",
        ("POST",   "sessions"):      "session_action",
        ("GET",    "admin"):         "admin_view",
        ("PATCH",  "admin"):         "admin_action",
        ("POST",   "auth/feedback"): "feedback",
    }

    for (m, prefix), label in mapping.items():
        if method == m and p.startswith(prefix):
            return label

    return f"{method.lower()}:{parts[0] if parts else path}"