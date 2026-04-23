from ultralytics import YOLO
from dotenv import load_dotenv
import torch
import os

# ── Load .env ─────────────────────────────────
load_dotenv()

YAML_PATH  = os.getenv("YAML_PATH")
RUNS_DIR   = os.getenv("RUNS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8s.pt")

EPOCHS          = int(os.getenv("EPOCHS",          50))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",       8))
IMG_SIZE        = int(os.getenv("IMG_SIZE",        640))
DEVICE          = int(os.getenv("DEVICE",            0))
PATIENCE        = int(os.getenv("PATIENCE",         15))
WORKERS         = int(os.getenv("WORKERS",           4))
LR0             = float(os.getenv("LR0",          0.001))
LRF             = float(os.getenv("LRF",          0.01))
MOMENTUM        = float(os.getenv("MOMENTUM",     0.937))
WEIGHT_DECAY    = float(os.getenv("WEIGHT_DECAY", 0.0005))
WARMUP_EPOCHS   = float(os.getenv("WARMUP_EPOCHS",  3.0))
WARMUP_MOMENTUM = float(os.getenv("WARMUP_MOMENTUM",0.8))

HSV_H     = float(os.getenv("HSV_H",     0.015))
HSV_S     = float(os.getenv("HSV_S",     0.7))
HSV_V     = float(os.getenv("HSV_V",     0.4))
FLIPLR    = float(os.getenv("FLIPLR",    0.5))
MOSAIC    = float(os.getenv("MOSAIC",    1.0))
MIXUP     = float(os.getenv("MIXUP",     0.1))
DEGREES   = float(os.getenv("DEGREES",  10.0))
TRANSLATE = float(os.getenv("TRANSLATE", 0.1))
SCALE     = float(os.getenv("SCALE",     0.5))

# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data            = YAML_PATH,
        epochs          = EPOCHS,
        imgsz           = IMG_SIZE,
        batch           = BATCH_SIZE,
        device          = DEVICE,
        project         = RUNS_DIR,
        name            = "cardd_yolov8s",
        patience        = PATIENCE,
        workers         = WORKERS,
        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_momentum = WARMUP_MOMENTUM,
        box             = 7.5,
        cls             = 0.5,
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,
        fliplr          = FLIPLR,
        mosaic          = MOSAIC,
        mixup           = MIXUP,
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        plots           = True,
        save            = True,
        verbose         = True
    )

    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {RUNS_DIR}/cardd_yolov8s")


if __name__ == "__main__":
    main()