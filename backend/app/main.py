from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
import os

from app.database import engine, Base
from app.routes.detection import router as detection_router
from app.routes.sessions  import router as sessions_router
from app.services.yolo_service import load_model

load_dotenv()

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Create tables ──────────────────────────────────────────────────────────
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

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(detection_router)
app.include_router(sessions_router)

# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting AUTOPS API...")
    print("📥 Loading YOLOv8 model...")
    load_model()
    print("✅ API Ready!")

# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}