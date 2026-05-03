from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.database import engine, Base
from app.routes.detection import router as detection_router
from app.services.yolo_service import load_model

# ── Load .env ─────────────────────────────────
load_dotenv()

# ── Create tables ─────────────────────────────
Base.metadata.create_all(bind=engine)

# ── App ───────────────────────────────────────
app = FastAPI(
    title       = "Car Damage Detection API",
    description = "Vision-based car damage detection using YOLOv8",
    version     = "1.0.0"
)

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ────────────────────────────────────
app.include_router(detection_router)

# ── Startup ───────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Car Damage Detection API...")
    print("📥 Loading YOLOv8 model...")
    load_model()
    print("✅ API Ready!")

# ── Health Check ──────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":  "healthy",
        "version": "1.0.0"
    }