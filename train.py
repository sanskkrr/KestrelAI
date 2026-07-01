from ultralytics import YOLO

def train():
    # Load pretrained YOLOv8 nano (smallest + fastest, good for CPU)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="config.yaml",        # your dataset config
        epochs=20,                 # keep low for now, increase later
        imgsz=640,                 # input image size
        batch=8,                   # lower if you get memory errors
        name="kestrel_baseline",   # run name
        project="models",          # saves to models/kestrel_baseline/
        patience=5,                # stop early if no improvement
        save=True,
        plots=True,                # saves training graphs
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Best weights saved to: models/kestrel_baseline/weights/best.pt")

if __name__ == "__main__":
    train()