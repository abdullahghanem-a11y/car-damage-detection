from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from dotenv import load_dotenv
import numpy as np
import base64
import cv2
import os

# ── Load .env ─────────────────────────────────
load_dotenv()

HF_REPO_ID     = os.getenv("HF_REPO_ID",     "abdullahg7/cardd-yolov8s")
MODEL_VERSION  = os.getenv("MODEL_VERSION",  "v1.0/best.pt")
MODEL_CACHE_DIR= os.getenv("MODEL_CACHE_DIR","./weights")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD",  0.45))
IMG_SIZE       = int(os.getenv("IMG_SIZE",         640))
_device_env = os.getenv("DEVICE", "0")
DEVICE = "cpu" if _device_env == "cpu" else int(_device_env)

CLASS_NAMES = [
    "dent", "scratch", "crack",
    "glass_shatter", "lamp_broken", "tire_flat"
]

CLASS_COLORS = {
    "dent":          (255, 99,  71),
    "scratch":       (30,  144, 255),
    "crack":         (50,  205, 50),
    "glass_shatter": (148, 0,   211),
    "lamp_broken":   (255, 215, 0),
    "tire_flat":     (255, 69,  0),
}

# ── Singleton model instance ───────────────────
_model = None


def load_model() -> YOLO:
    """Download from HuggingFace if not cached, then load."""
    global _model

    if _model is not None:
        return _model

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_CACHE_DIR, "best.pt")

    if not os.path.exists(local_path):
        print(f"📥 Downloading model from HuggingFace: {HF_REPO_ID}/{MODEL_VERSION}")
        downloaded = hf_hub_download(
            repo_id  = HF_REPO_ID,
            filename = MODEL_VERSION,
            cache_dir= MODEL_CACHE_DIR
        )
        # Copy to a clean path
        import shutil
        shutil.copy(downloaded, local_path)
        print(f"✅ Model cached at: {local_path}")
    else:
        print(f"✅ Model loaded from cache: {local_path}")

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    _model = YOLO(local_path)
    return _model


def run_inference(image_bytes: bytes, filename: str) -> dict:
    """Run YOLOv8 inference on image bytes."""
    model = load_model()

    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not decode image: {filename}")

    # Run inference
    results = model.predict(
        source  = img,
        imgsz   = IMG_SIZE,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        device  = DEVICE,
        verbose = False
    )[0]

    # Parse detections
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id   = int(box.cls)
            class_name = CLASS_NAMES[class_id]
            confidence = float(box.conf)

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox":       [round(x1, 2), round(y1, 2),
                               round(x2, 2), round(y2, 2)]
            })

            # Draw bounding box on image
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(img, (int(x1), int(y1)),
                               (int(x2), int(y2)), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label,
                       (int(x1), int(y1) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

    # Encode annotated image to base64
    _, buffer       = cv2.imencode(".jpg", img)
    annotated_b64   = base64.b64encode(buffer).decode("utf-8")

    return {
        "filename":        filename,
        "total_instances": len(detections),
        "detections":      detections,
        "annotated_image": annotated_b64
    }