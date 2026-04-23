import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────
load_dotenv()

IMAGES_DIR = os.getenv("IMAGES_DIR")
LABELS_DIR = os.getenv("LABELS_DIR")
SPLITS     = ["train", "val", "test"]

# ──────────────────────────────────────────────

print("=" * 50)
print("  CarDD Dataset Sanity Check")
print("=" * 50)

for split in SPLITS:
    img_dir   = Path(IMAGES_DIR) / split
    label_dir = Path(LABELS_DIR) / split

    images = set(p.stem for p in img_dir.glob("*")
                 if p.suffix.lower() in [".jpg", ".jpeg", ".png"])

    labels = set(p.stem for p in label_dir.glob("*.txt"))

    matched  = images & labels
    no_label = images - labels
    no_image = labels - images

    print(f"\n📂 {split.upper()} split:")
    print(f"   Images found   : {len(images)}")
    print(f"   Labels found   : {len(labels)}")
    print(f"   Matched pairs  : {len(matched)} ✅")

    if no_label:
        print(f"   ⚠️  Images without labels: {len(no_label)}")
    if no_image:
        print(f"   ⚠️  Labels without images: {len(no_image)}")

print("\n" + "=" * 50)
print("  Sanity check complete!")
print("=" * 50)