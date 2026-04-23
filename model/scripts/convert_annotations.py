import json
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────
load_dotenv()

ANNOTATIONS_DIR = os.getenv("ANNOTATIONS_DIR")
LABELS_DIR      = os.getenv("LABELS_DIR")
SPLITS          = ["train", "val", "test"]

# ──────────────────────────────────────────────

def convert_coco_to_yolo(json_path, output_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build lookup dictionaries
    image_info = {
        img["id"]: {
            "file_name": img["file_name"],
            "width":     img["width"],
            "height":    img["height"]
        }
        for img in data["images"]
    }

    # Group annotations by image id
    annotations_per_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []
        annotations_per_image[img_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    skipped   = 0
    empty     = 0

    for img_id, img_data in tqdm(image_info.items(), desc=f"Converting {Path(json_path).stem}"):
        w = img_data["width"]
        h = img_data["height"]

        base_name  = Path(img_data["file_name"]).stem
        label_path = os.path.join(output_dir, base_name + ".txt")
        anns       = annotations_per_image.get(img_id, [])

        if not anns:
            open(label_path, "w").close()
            empty += 1
            continue

        with open(label_path, "w") as f:
            for ann in anns:
                if ann.get("iscrowd", 0):
                    skipped += 1
                    continue

                cat_id     = ann["category_id"] - 1
                x, y, bw, bh = ann["bbox"]

                if bw <= 0 or bh <= 0:
                    skipped += 1
                    continue

                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                converted += 1

    print(f"\n✅ Done: {converted} boxes converted | {empty} empty images | {skipped} skipped")


if __name__ == "__main__":
    for split in SPLITS:
        json_path  = os.path.join(ANNOTATIONS_DIR, f"{split}.json")
        output_dir = os.path.join(LABELS_DIR, split)

        if not os.path.exists(json_path):
            print(f"⚠️  Skipping {split} — {json_path} not found")
            continue

        print(f"\n📂 Processing {split} split...")
        convert_coco_to_yolo(json_path, output_dir)

    print("\n🎉 All splits converted successfully!")