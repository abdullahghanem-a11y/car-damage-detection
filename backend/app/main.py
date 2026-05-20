from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import os

from app.database import engine, Base, SessionLocal
from app.routes.detection import router as detection_router
from app.routes.sessions  import router as sessions_router
from app.routes.auth      import router as auth_router, ensure_admin_exists
from app.routes.admin     import router as admin_router
from app.middleware.logging_middleware import ActivityLogMiddleware
from app.services.yolo_service import load_model

load_dotenv()

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Create all tables ──────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AUTOPS — Car Damage Detection API",
    description = "Vision-based car damage detection using YOLOv8",
    version     = "2.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Activity logging middleware ────────────────────────────────────────────
app.add_middleware(ActivityLogMiddleware)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(sessions_router)
app.include_router(admin_router)

# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting AUTOPS API...")
    db = SessionLocal()
    try:
        ensure_admin_exists(db)
        print("✅ Admin account ready")
    finally:
        db.close()
    print("📥 Loading YOLOv8 model...")
    load_model()
    print("✅ API Ready!")


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}