from ultralytics import YOLO
from pathlib import Path
import shutil

MODEL_PATH = "models/final/baseline_best.pt"
UNLABELED_DIR = "data/raw/images"
OUT_IMG_DIR = "data/processed/images/train"      # we add pseudo-labels into existing train set
OUT_LBL_DIR = "data/processed/labels/train"
CONFIDENCE_THRESHOLD = 0.45

def generate_pseudo_labels():
    model = YOLO(MODEL_PATH)

    unlabeled_images = list(Path(UNLABELED_DIR).glob("*.jpg"))
    print(f"Found {len(unlabeled_images)} unlabeled images")

    Path(OUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_LBL_DIR).mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for i, img_path in enumerate(unlabeled_images):
        results = model(str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # Skip images where the model found nothing confident
        if len(result.boxes) == 0:
            skipped += 1
            continue

        # Convert detections to YOLO label format
        lines = []
        img_h, img_w = result.orig_shape

        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, w, h = box.xywh[0].tolist()

            # normalize
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Save pseudo-labeled image + label, prefixed so we can identify them later
        new_name = f"pseudo_{img_path.stem}"
        shutil.copy(img_path, f"{OUT_IMG_DIR}/{new_name}.jpg")

        with open(f"{OUT_LBL_DIR}/{new_name}.txt", "w") as f:
            f.write("\n".join(lines))

        kept += 1

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(unlabeled_images)}...")

    print(f"\n✅ Pseudo-labeling complete!")
    print(f"Kept: {kept} images (confident detections)")
    print(f"Skipped: {skipped} images (no confident detections)")
    print(f"New training set size: {len(list(Path(OUT_IMG_DIR).glob('*.jpg')))}")


if __name__ == "__main__":
    generate_pseudo_labels()