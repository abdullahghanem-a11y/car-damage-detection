from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Load .env ─────────────────────────────────
load_dotenv()

BEST_MODEL  = os.getenv("BEST_MODEL")
TEST_IMAGES = os.path.join(os.getenv("IMAGES_DIR"), "test")
OUTPUT_DIR  = os.path.join(os.getenv("RUNS_DIR"), "visualizations")
DEVICE      = int(os.getenv("DEVICE", 0))
CONF        = float(os.getenv("CONF_THRESHOLD", 0.25))

# ──────────────────────────────────────────────

COLORS = {
    "dent":          (255, 99,  71),
    "scratch":       (30,  144, 255),
    "crack":         (50,  205, 50),
    "glass_shatter": (148, 0,   211),
    "lamp_broken":   (255, 215, 0),
    "tire_flat":     (255, 69,  0),
}


def visualize_predictions(model_path, images_dir, output_dir, num_images=16):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    all_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected = random.sample(all_images, min(num_images, len(all_images)))

    print(f"Running inference on {len(selected)} images...")
    results = model.predict(
        source  = selected,
        imgsz   = int(os.getenv("IMG_SIZE", 640)),
        conf    = CONF,
        device  = DEVICE,
        verbose = False
    )

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        img = result.orig_img[:, :, ::-1]
        axes[idx].imshow(img)
        axes[idx].axis("off")

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = result.names[int(box.cls)]
                conf     = float(box.conf)
                color    = [c / 255 for c in COLORS.get(cls_name, (255, 255, 255))]

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                axes[idx].add_patch(rect)
                axes[idx].text(
                    x1, y1 - 5,
                    f"{cls_name} {conf:.2f}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1)
                )

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("YOLOv8s — Car Damage Detection Results", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "predictions_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    visualize_predictions(BEST_MODEL, TEST_IMAGES, OUTPUT_DIR, num_images=16)