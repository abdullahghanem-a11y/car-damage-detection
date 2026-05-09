from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import warnings
import base64
import shutil
import torch
import cv2
import os

# ── Load .env ─────────────────────────────────
load_dotenv()

HF_REPO_ID      = os.getenv("HF_REPO_ID",      "abdullahg7/cardd-yolov8s")
MODEL_VERSION   = os.getenv("MODEL_VERSION",   "v1.0/best.pt")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./weights")
CONF_THRESHOLD  = float(os.getenv("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD   = float(os.getenv("IOU_THRESHOLD",  0.45))
IMG_SIZE        = int(os.getenv("IMG_SIZE",         640))
DEVICE_ENV      = os.getenv("DEVICE", "0")
DEVICE          = "cpu" if DEVICE_ENV == "cpu" else int(DEVICE_ENV)

# ── Car Re-identification threshold ───────────
SIMILARITY_THRESHOLD = 0.80

# ── Accepted vehicle classes from COCO ────────
ACCEPTED_VEHICLES = {"car"}

# ── Damage class names ────────────────────────
CLASS_NAMES = [
    "dent", "scratch", "crack",
    "glass_shatter", "lamp_broken", "tire_flat"
]

# ── Damage class colors ───────────────────────
CLASS_COLORS = {
    "dent":          (255, 99,  71),
    "scratch":       (30,  144, 255),
    "crack":         (50,  205, 50),
    "glass_shatter": (148, 0,   211),
    "lamp_broken":   (255, 215, 0),
    "tire_flat":     (255, 69,  0),
}

# ── Severity Config ───────────────────────────
# Damage type severity weights (higher = more severe)
DAMAGE_WEIGHTS = {
    "crack":         1.0,
    "glass_shatter": 0.95,
    "dent":          0.75,
    "lamp_broken":   0.70,
    "scratch":       0.50,
    "tire_flat":     0.85,
}

# ── Singleton model instances ─────────────────
_damage_model = None   # our fine-tuned CarDD model
_car_model    = None   # COCO pretrained for car verification
_feature_model = None  # ResNet for re-identification
_transform    = None   # image transform for ResNet


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_damage_model() -> YOLO:
    """Load fine-tuned CarDD damage detection model."""
    global _damage_model
    if _damage_model is not None:
        return _damage_model

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_CACHE_DIR, "best.pt")

    if not os.path.exists(local_path):
        print(f"📥 Downloading damage model from HuggingFace: {HF_REPO_ID}/{MODEL_VERSION}")
        downloaded = hf_hub_download(
            repo_id  = HF_REPO_ID,
            filename = MODEL_VERSION,
            cache_dir= MODEL_CACHE_DIR
        )
        shutil.copy(downloaded, local_path)
        print(f"✅ Damage model cached at: {local_path}")
    else:
        print(f"✅ Damage model loaded from cache: {local_path}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    _damage_model = YOLO(local_path)
    return _damage_model


def load_car_model() -> YOLO:
    """Load COCO pretrained YOLOv8 for car verification."""
    global _car_model
    if _car_model is not None:
        return _car_model

    print("📥 Loading COCO YOLOv8 for car verification...")
    _car_model = YOLO("yolov8s.pt")
    print("✅ Car verification model ready!")
    return _car_model


def load_feature_model():
    """Load ResNet18 for car re-identification feature extraction."""
    global _feature_model, _transform
    if _feature_model is not None:
        return _feature_model, _transform

    print("📥 Loading ResNet18 for car re-identification...")
    _feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove final classification layer — we want features only
    _feature_model = torch.nn.Sequential(*list(_feature_model.children())[:-1])
    _feature_model.eval()

    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ])
    print("✅ Re-identification model ready!")
    return _feature_model, _transform


def load_model():
    """Load all models at startup."""
    load_damage_model()
    load_car_model()
    load_feature_model()


# ─────────────────────────────────────────────
# Stage 1 — Car Verification
# ─────────────────────────────────────────────

def verify_car(img: np.ndarray) -> dict:
    """
    Check if image contains a car using COCO pretrained YOLOv8.
    Returns verification result with detected vehicle type.
    """
    car_model = load_car_model()

    results = car_model.predict(
        source  = img,
        imgsz   = 640,
        conf    = 0.3,
        verbose = False
    )[0]

    detected_vehicles = []
    if results.boxes is not None:
        for box in results.boxes:
            class_name = car_model.names[int(box.cls)].lower()
            confidence = float(box.conf)
            if class_name in ACCEPTED_VEHICLES:
                detected_vehicles.append({
                    "type":       class_name,
                    "confidence": round(confidence, 4)
                })

    if detected_vehicles:
        # Return the most confident vehicle detection
        best = max(detected_vehicles, key=lambda x: x["confidence"])
        return {
            "is_car":       True,
            "vehicle_type": best["type"],
            "confidence":   best["confidence"],
            "message":      f"{best['type'].capitalize()} detected with {best['confidence']*100:.1f}% confidence"
        }
    else:
        return {
            "is_car":       False,
            "vehicle_type": None,
            "confidence":   0.0,
            "message":      "No car detected in this image. Please upload a car image."
        }


# ─────────────────────────────────────────────
# Stage 2 — Car Re-identification
# ─────────────────────────────────────────────

def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract visual feature vector from image using ResNet18."""
    feature_model, transform = load_feature_model()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        features = feature_model(tensor)

    return features.squeeze().numpy().flatten()


def extract_dominant_color(img: np.ndarray) -> np.ndarray:
    """Extract dominant color of the car using color histogram."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize for speed
    small   = cv2.resize(img_rgb, (64, 64))
    # Calculate mean color in HSV space
    img_hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mean_color = np.mean(img_hsv.reshape(-1, 3), axis=0)
    return mean_color


def group_images_by_car(images: list) -> dict:
    """
    Group images that show the same car using feature similarity.

    Returns:
        dict with groups of same-car images and rejected images
    """
    if len(images) == 1:
        return {
            "groups":   [{"images": [images[0]], "representative": 0}],
            "rejected": []
        }

    # Extract features for all images
    features     = []
    color_vectors = []
    for img_data in images:
        feat  = extract_features(img_data["img"])
        color = extract_dominant_color(img_data["img"])
        features.append(feat)
        color_vectors.append(color)

    features      = np.array(features)
    color_vectors = np.array(color_vectors)

    # Normalize features
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    # Calculate pairwise cosine similarity
    sim_matrix = cosine_similarity(features)

    # Group images by similarity
    n       = len(images)
    visited = [False] * n
    groups  = []

    for i in range(n):
        if visited[i]:
            continue
        group   = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                group.append(j)
                visited[j] = True
        groups.append(group)

    # Build result
    result_groups = []
    for group in groups:
        result_groups.append({
            "images":          [images[idx] for idx in group],
            "representative":  group[0],
            "size":            len(group)
        })

    return {
        "groups":   result_groups,
        "rejected": []
    }


# ─────────────────────────────────────────────
# Stage 3 — Severity Scoring
# ─────────────────────────────────────────────

def calculate_severity(
    confidence: float,
    bbox:       list,
    img_shape:  tuple,
    class_name: str
) -> dict:
    """
    Calculate damage severity based on:
    - Confidence score
    - Bounding box area relative to image
    - Damage type weight
    """
    img_h, img_w = img_shape[:2]
    img_area     = img_w * img_h

    # Bounding box area ratio
    x1, y1, x2, y2 = bbox
    box_area  = (x2 - x1) * (y2 - y1)
    area_ratio = box_area / img_area

    # Damage type weight
    type_weight = DAMAGE_WEIGHTS.get(class_name, 0.7)

    # Severity score (0-1)
    score = (confidence * 0.4) + (area_ratio * 0.3) + (type_weight * 0.3)
    score = min(1.0, score)

    # Map to label
    if score < 0.35:
        label = "Minor"
        color = "#22c55e"    # green
    elif score < 0.65:
        label = "Moderate"
        color = "#f59e0b"    # yellow
    else:
        label = "Severe"
        color = "#ef4444"    # red

    return {
        "score": round(score, 4),
        "label": label,
        "color": color
    }


# ─────────────────────────────────────────────
# Core Inference
# ─────────────────────────────────────────────

def run_inference(image_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline:
    1. Verify car presence
    2. Run damage detection
    3. Calculate severity scores
    """
    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not decode image: {filename}")

    # ── Stage 1: Car Verification ──────────────
    verification = verify_car(img)

    if not verification["is_car"]:
        return {
            "filename":        filename,
            "verified":        False,
            "vehicle_type":    None,
            "rejection_reason": verification["message"],
            "total_instances": 0,
            "detections":      [],
            "annotated_image": _encode_image(img),
            "overall_severity": None
        }

    # ── Stage 2: Damage Detection ──────────────
    damage_model = load_damage_model()
    results = damage_model.predict(
        source  = img,
        imgsz   = IMG_SIZE,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        device  = DEVICE,
        verbose = False
    )[0]

    # ── Stage 3: Parse + Severity Scoring ──────
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id        = int(box.cls)
            class_name      = CLASS_NAMES[class_id]
            confidence      = float(box.conf)
            bbox            = [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]

            severity = calculate_severity(
                confidence = confidence,
                bbox       = bbox,
                img_shape  = img.shape,
                class_name = class_name
            )

            detections.append({
                "class_id":        class_id,
                "class_name":      class_name,
                "confidence":      round(confidence, 4),
                "bbox":            bbox,
                "severity_label":  severity["label"],
                "severity_score":  severity["score"],
                "severity_color":  severity["color"],
            })

            # Draw bounding box
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(img,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)), color, 2)

            # Draw label with severity
            label = f"{class_name} {confidence:.2f} [{severity['label']}]"
            cv2.putText(img, label,
                       (int(x1), int(y1) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, color, 2)

    # ── Overall Severity ───────────────────────
    overall_severity = _calculate_overall_severity(detections)

    return {
        "filename":         filename,
        "verified":         True,
        "vehicle_type":     verification["vehicle_type"],
        "rejection_reason": None,
        "total_instances":  len(detections),
        "detections":       detections,
        "annotated_image":  _encode_image(img),
        "overall_severity": overall_severity
    }


def run_batch_inference(images_data: list) -> dict:
    """
    Multi-angle pipeline:
    1. Verify each image has a car
    2. Group verified images by car identity
    3. Run damage detection on each group
    4. Aggregate results per car
    """
    verified   = []
    rejected   = []

    # Stage 1 — Verify all images
    for item in images_data:
        np_arr = np.frombuffer(item["bytes"], np.uint8)
        img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            rejected.append({
                "filename": item["filename"],
                "reason":   "Could not decode image"
            })
            continue

        verification = verify_car(img)
        if verification["is_car"]:
            verified.append({
                "filename":     item["filename"],
                "bytes":        item["bytes"],
                "img":          img,
                "vehicle_type": verification["vehicle_type"]
            })
        else:
            rejected.append({
                "filename": item["filename"],
                "reason":   verification["message"]
            })

    if not verified:
        return {
            "total_images":    len(images_data),
            "verified_images": 0,
            "rejected_images": rejected,
            "car_groups":      [],
            "total_instances": 0
        }

    # Stage 2 — Group by car identity
    grouping = group_images_by_car(verified)

    # Stage 3 — Run damage detection per group
    car_groups = []
    total_instances = 0

    for group in grouping["groups"]:
        group_results   = []
        group_detections = []

        for img_data in group["images"]:
            result = run_inference(img_data["bytes"], img_data["filename"])
            group_results.append(result)
            group_detections.extend(result["detections"])
            total_instances += result["total_instances"]

        # Aggregate detections across all angles
        aggregated = _aggregate_detections(group_detections)

        car_groups.append({
            "vehicle_type":        group["images"][0]["vehicle_type"],
            "image_count":         len(group["images"]),
            "images":              group_results,
            "aggregated_damage":   aggregated,
            "total_instances":     len(group_detections),
            "overall_severity":    _calculate_overall_severity(group_detections)
        })

    return {
        "total_images":    len(images_data),
        "verified_images": len(verified),
        "rejected_images": rejected,
        "car_groups":      car_groups,
        "total_instances": total_instances
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _encode_image(img: np.ndarray) -> str:
    """Encode image to base64 string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def _calculate_overall_severity(detections: list) -> dict:
    """Calculate overall severity from all detections."""
    if not detections:
        return {
            "label": "None",
            "score": 0.0,
            "color": "#64748b"
        }

    avg_score = np.mean([d["severity_score"] for d in detections])

    if avg_score < 0.35:
        label = "Minor"
        color = "#22c55e"
    elif avg_score < 0.65:
        label = "Moderate"
        color = "#f59e0b"
    else:
        label = "Severe"
        color = "#ef4444"

    return {
        "label": label,
        "score": round(float(avg_score), 4),
        "color": color
    }


def _aggregate_detections(detections: list) -> list:
    """
    Aggregate detections across multiple images of the same car.
    Groups by damage class and keeps the most severe instance of each.
    """
    if not detections:
        return []

    # Group by class name
    by_class = {}
    for det in detections:
        cls = det["class_name"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)

    # Keep most severe instance of each class
    aggregated = []
    for cls, dets in by_class.items():
        most_severe = max(dets, key=lambda x: x["severity_score"])
        aggregated.append({
            "class_name":      cls,
            "count":           len(dets),
            "severity_label":  most_severe["severity_label"],
            "severity_score":  most_severe["severity_score"],
            "severity_color":  most_severe["severity_color"],
            "max_confidence":  max(d["confidence"] for d in dets)
        })

    # Sort by severity score descending
    aggregated.sort(key=lambda x: x["severity_score"], reverse=True)
    return aggregated