from ultralytics import YOLO
from dotenv import load_dotenv
import os

# ── Load .env ─────────────────────────────────
load_dotenv()

BEST_MODEL = os.getenv("BEST_MODEL")
YAML_PATH  = os.getenv("YAML_PATH")
DEVICE     = int(os.getenv("DEVICE", 0))
IMG_SIZE   = int(os.getenv("IMG_SIZE", 640))

# ──────────────────────────────────────────────

model = YOLO(BEST_MODEL)

if __name__ == "__main__":
    metrics = model.val(
        data    = YAML_PATH,
        split   = "test",
        imgsz   = IMG_SIZE,
        device  = DEVICE,
        plots   = True,
        verbose = True
    )

    print("\n" + "=" * 50)
    print("  FINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("=" * 50)