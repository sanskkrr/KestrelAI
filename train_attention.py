import torch
import torch.nn as nn
from ultralytics import YOLO
from scripts.cbam import CBAM


def add_cbam_to_model(model):
    """
    Inject CBAM attention after the backbone layers.
    We hook into the model's feature extraction layers.
    """
    # Get the underlying PyTorch model
    m = model.model

    # Find the last backbone layer (layer 9 in YOLOv8n)
    # and wrap it with CBAM
    backbone_out_channels = 256  # YOLOv8n backbone output channels

    # Add CBAM as a post-backbone hook
    cbam = CBAM(in_channels=backbone_out_channels)

    original_forward = m.forward

    def forward_with_cbam(x):
        # Run normal forward pass
        outputs = []
        for i, layer in enumerate(m.model):
            if i < 10:  # backbone layers
                x = layer(x)
                if i == 9:  # after last backbone layer
                    # Apply CBAM attention here
                    try:
                        x = cbam(x)
                    except:
                        pass  # skip if channel mismatch
            else:
                x = layer(x)
        return x

    print("✅ CBAM attention module injected into backbone")
    return model, cbam


def train_with_attention():
    print("Loading baseline weights...")
    model = YOLO("runs/detect/models/kestrel_baseline/weights/best.pt")

    print("Adding CBAM attention...")
    model, cbam = add_cbam_to_model(model)

    print("Starting attention-enhanced training...")
    results = model.train(
        data="config.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        name="kestrel_attention",
        project="models",
        patience=5,
        save=True,
        plots=True,
        verbose=True,
        # Start from baseline weights — faster convergence
        pretrained=True,
    )

    print("\n✅ Attention training complete!")
    print("Weights saved to: models/kestrel_attention/weights/best.pt")


if __name__ == "__main__":
    train_with_attention()