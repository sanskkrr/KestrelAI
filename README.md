# 🦅 KestrelAI

> UAV object detection system trained on real drone footage. Upload an aerial image, get bounding boxes and attention heatmaps back in under 100ms.

Built on YOLOv8 with a pseudo-labeling pipeline that expands labeled training data automatically — no manual annotation needed beyond the seed dataset.

---

## what it does

- detects 10 object classes in drone imagery: pedestrians, cars, trucks, vans, buses, bicycles, tricycles, motorcycles, and more
- generates Grad-CAM heatmaps showing exactly where the model focused
- runs inference on both images and videos
- adjustable confidence threshold via sidebar slider
- 30.8% mAP@50 on VisDrone 2019 validation set

---

## how it was built

**dataset** — VisDrone 2019 (6,382 annotated aerial images across 10 classes). Annotations converted from VisDrone format to YOLO format via a custom conversion script.

**baseline** — YOLOv8n pretrained on COCO, fine-tuned on VisDrone for 50 epochs. Established the detection foundation at 28.3% mAP@50.

**attention** — CBAM (Convolutional Block Attention Module) integrated directly into YOLOv8's backbone architecture via a custom model YAML, inserted after the SPPF layer to refine feature maps before the detection head. Result: comparable performance to baseline (28.1% mAP@50), with the finding that single-point attention at low-resolution feature stages provides minimal gain for small object detection — a documented research finding, not a failure.

**pseudo-labeling** — the trained baseline model ran inference on 1,610 unlabeled VisDrone test-dev images, auto-generating labels for 1,584 of them (confidence threshold: 0.45). These pseudo-labeled images were merged with the original training set, expanding it by ~25% without any manual annotation.

**final model** — retrained from baseline weights on the expanded dataset (7,966 images). Final mAP@50: **30.8%**, a +2.5 point improvement over baseline, demonstrating the value of semi-supervised learning for aerial data scarcity.

---

## stack

```
model         YOLOv8n (Ultralytics)
attention     CBAM — channel + spatial attention
framework     PyTorch
cv tools      OpenCV, Albumentations
backend       FastAPI + Uvicorn
frontend      Streamlit
dataset       VisDrone 2019
language      Python 3.12
```

---

## project structure

```
KestrelAI/
├── models/
│   └── final/
│       ├── baseline_best.pt       # baseline model weights
│       └── kestrel_final_best.pt  # final model weights (use this)
├── scripts/
│   ├── convert_visdrone.py        # VisDrone → YOLO format converter
│   ├── pseudo_label.py            # auto-label unlabeled drone images
│   └── gradcam.py                 # generate attention heatmaps
├── results/
│   ├── detections/                # saved detection outputs
│   └── heatmaps/                  # saved Grad-CAM outputs
├── app.py                         # Streamlit web app
├── train.py                       # baseline training script
├── train_attention.py             # CBAM attention training
├── predict.py                     # run inference on single image
└── config.yaml                    # dataset config for YOLO
```

---

## running locally

```bash
# clone
git clone https://github.com/yourusername/KestrelAI.git
cd KestrelAI

# setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the app
streamlit run app.py
```

open `http://localhost:8501`, upload any drone image or video.

---

## results

| model | training images | epochs | mAP@50 |
|---|---|---|---|
| baseline YOLOv8n | 6,382 | 50 | 28.3% |
| + CBAM attention | 6,382 | 50 | 28.1% |
| + pseudo-labeling | 7,966 | 50 | **30.8%** |

per-class breakdown (final model):

| class | mAP@50 |
|---|---|
| car | 73.3% |
| bus | 41.6% |
| van | 35.5% |
| pedestrian | 31.6% |
| truck | 27.9% |
| motor | 34.6% |
| tricycle | 20.8% |
| people | 24.5% |
| awning-tricycle | 11.0% |
| bicycle | 6.97% |

cars and buses perform well due to high instance counts in training data. small/rare classes (bicycle: 1,287 instances vs car: 14,064) are the primary drag on overall mAP — a class imbalance problem, not a pipeline issue.

---

## key findings

pseudo-labeling gave a clean +2.5% mAP improvement with zero manual annotation effort — the model essentially taught itself using its own confident predictions on unseen data.

CBAM attention, when inserted only at the lowest-resolution backbone stage, doesn't meaningfully improve small object detection. multi-scale attention placement at higher-resolution P3/P4 feature maps is the likely path to actual improvement, and remains an open direction.

---

## dataset

[VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) — Task 1: Object Detection in Images

not included in this repo due to size. download train + val sets and run `scripts/convert_visdrone.py` to prepare the data.

---

made by Sanskar Chouksey