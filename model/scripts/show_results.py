import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dotenv import load_dotenv
import os

# ── Load .env ─────────────────────────────────
load_dotenv()

RUNS_DIR   = os.getenv("RUNS_DIR")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR")

# Derive the run folder from weights dir (one level up)
RUN_DIR  = os.path.dirname(WEIGHTS_DIR)
OUTPUT   = os.path.join(RUNS_DIR, "visualizations")
os.makedirs(OUTPUT, exist_ok=True)

# ──────────────────────────────────────────────

# ── 1. Training Curves ─────────────────────────
img = mpimg.imread(os.path.join(RUN_DIR, "results.png"))
plt.figure(figsize=(18, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Training Curves", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Training curves saved")

# ── 2. Confusion Matrix ────────────────────────
img = mpimg.imread(os.path.join(RUN_DIR, "confusion_matrix_normalized.png"))
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis("off")
plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Confusion matrix saved")

# ── 3. PR Curve ────────────────────────────────
img = mpimg.imread(os.path.join(RUN_DIR, "PR_curve.png"))
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis("off")
plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "pr_curve.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ PR curve saved")

# ── 4. F1 Curve ────────────────────────────────
img = mpimg.imread(os.path.join(RUN_DIR, "F1_curve.png"))
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis("off")
plt.title("F1 Curve", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "f1_curve.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ F1 curve saved")

# ── 5. Per-Class Bar Chart ─────────────────────
classes     = ["Glass\nShatter", "Tire\nFlat", "Lamp\nBroken", "Scratch", "Dent", "Crack"]
yolov8_map  = [0.990, 0.905, 0.855, 0.596, 0.578, 0.377]
dcnplus_map = [0.926, 0.860, 0.708, 0.343, 0.405, 0.166]

x     = range(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar([i - width/2 for i in x], yolov8_map,  width, label="YOLOv8s (Ours)",  color="#2196F3", alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], dcnplus_map, width, label="DCN+ ResNet-101", color="#FF5722", alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2196F3")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#FF5722")

ax.set_xlabel("Damage Category", fontsize=12)
ax.set_ylabel("mAP50", fontsize=12)
ax.set_title("Per-Class mAP50: YOLOv8s vs DCN+ ResNet-101", fontsize=14, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "perclass_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Per-class comparison saved")

# ── 6. Val Predictions vs Ground Truth ─────────
fig, axes = plt.subplots(3, 2, figsize=(18, 22))
pairs = [
    ("val_batch0_labels.jpg", "val_batch0_pred.jpg"),
    ("val_batch1_labels.jpg", "val_batch1_pred.jpg"),
    ("val_batch2_labels.jpg", "val_batch2_pred.jpg"),
]
for row, (label_file, pred_file) in enumerate(pairs):
    label_img = mpimg.imread(os.path.join(RUN_DIR, label_file))
    pred_img  = mpimg.imread(os.path.join(RUN_DIR, pred_file))
    axes[row][0].imshow(label_img)
    axes[row][0].axis("off")
    axes[row][0].set_title(f"Ground Truth — Batch {row}", fontsize=12, fontweight="bold")
    axes[row][1].imshow(pred_img)
    axes[row][1].axis("off")
    axes[row][1].set_title(f"Predictions — Batch {row}", fontsize=12, fontweight="bold")

plt.suptitle("Ground Truth vs Predictions on Validation Set", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "gt_vs_pred.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ GT vs Predictions saved")

print("\n🎉 All figures saved to:", OUTPUT)