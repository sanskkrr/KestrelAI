import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/final/kestrel_final_best.pt"
OUTPUT_DIR = "results/heatmaps"

def generate_gradcam(img_path, output_name="gradcam_output.jpg"):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(MODEL_PATH)

    # Run prediction with visualize=True — Ultralytics saves feature maps
    results = model(
        img_path,
        visualize=True,
        project=OUTPUT_DIR,
        name="gradcam_run",
        exist_ok=True,
        save=True,
        conf=0.35
    )

    print(f"✅ Feature maps saved to {OUTPUT_DIR}/gradcam_run/")
    print(f"Detected {len(results[0].boxes)} objects")

    # Also generate a simple activation heatmap manually
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Get prediction with boxes drawn
    annotated = results[0].plot()

    # Create a simple confidence heatmap overlay
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf)
        # Draw gaussian blob centered on each detection
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for y in range(max(0, cy-40), min(h, cy+40)):
            for x in range(max(0, cx-40), min(w, cx+40)):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                heatmap[y, x] += conf * np.exp(-dist**2 / (2 * 20**2))

    # Normalize and colorize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    # Add detection boxes on top
    final = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)

    output_path = f"{OUTPUT_DIR}/{output_name}"
    cv2.imwrite(output_path, final)
    print(f"✅ Heatmap saved to {output_path}")


if __name__ == "__main__":
    # Run on a val image
    import os
    val_dir = "data/processed/images/val"
    img = os.listdir(val_dir)[0]
    generate_gradcam(f"{val_dir}/{img}")