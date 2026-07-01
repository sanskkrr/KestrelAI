import os
import shutil
import cv2
from pathlib import Path

CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

def convert_annotation(ann_path, img_w, img_h):
    yolo_lines = []
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            category = int(parts[5])

            if score == 0 or category == 0 or category not in CATEGORY_MAP:
                continue

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            cls = CATEGORY_MAP[category]
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def process_split(split_name, raw_dir, out_dir):
    img_in  = Path(raw_dir) / f"VisDrone2019-DET-{split_name}" / "images"
    ann_in  = Path(raw_dir) / f"VisDrone2019-DET-{split_name}" / "annotations"
    img_out = Path(out_dir) / "images" / split_name
    lbl_out = Path(out_dir) / "labels" / split_name

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = sorted(img_in.glob("*.jpg"))
    print(f"\n[{split_name}] Found {len(images)} images")

    skipped = 0
    for i, img_path in enumerate(images):
        ann_path = ann_in / (img_path.stem + ".txt")

        if not ann_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        if not yolo_lines:
            skipped += 1
            continue

        shutil.copy(img_path, img_out / img_path.name)

        with open(lbl_out / (img_path.stem + ".txt"), 'w') as f:
            f.write("\n".join(yolo_lines))

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(images)}...")

    processed = len(images) - skipped
    print(f"[{split_name}] Done — Processed: {processed} | Skipped: {skipped}")


if __name__ == "__main__":
    RAW_DIR = "data/raw"
    OUT_DIR = "data/processed"

    print("Starting VisDrone → YOLO conversion...")
    process_split("train", RAW_DIR, OUT_DIR)
    process_split("val",   RAW_DIR, OUT_DIR)
    print("\n✅ Conversion complete!")
    print(f"Images → {OUT_DIR}/images/")
    print(f"Labels → {OUT_DIR}/labels/")