from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("models/final/kestrel_final_best.pt")

# Run on a val image
import os
val_images = os.listdir("data/processed/images/val")
test_img = f"data/processed/images/val/{val_images[0]}"

results = model(test_img)
results[0].save("results/detections/final_test.jpg")
print(f"✅ Saved to results/detections/final_test.jpg")
print(f"Detected {len(results[0].boxes)} objects")