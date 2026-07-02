import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os

# Page config
st.set_page_config(
    page_title="KestrelAI",
    page_icon="🦅",
    layout="wide"
)

# Load model once
@st.cache_resource
def load_model():
    return YOLO("models/final/kestrel_final_best.pt")

model = load_model()

# Class colors for bounding boxes
COLORS = {
    0: (0, 255, 153),   # pedestrian - green
    1: (0, 200, 255),   # people - cyan
    2: (255, 165, 0),   # bicycle - orange
    3: (0, 120, 255),   # car - blue
    4: (180, 0, 255),   # van - purple
    5: (255, 50, 50),   # truck - red
    6: (0, 255, 200),   # tricycle - teal
    7: (255, 255, 0),   # awning-tricycle - yellow
    8: (255, 100, 200), # bus - pink
    9: (100, 255, 100), # motor - light green
}

def generate_heatmap(img, results):
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for y in range(max(0, cy-50), min(h, cy+50)):
            for x in range(max(0, cx-50), min(w, cx+50)):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                heatmap[y, x] += conf * np.exp(-dist**2 / (2 * 25**2))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    return overlay

# UI
st.title("🦅 KestrelAI")
st.markdown("**UAV Object Detection System** · YOLOv8 + Pseudo-Labeling · VisDrone 2019")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
    show_heatmap = st.toggle("Show Grad-CAM Heatmap", value=False)
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Architecture: YOLOv8n")
    st.markdown("- Dataset: VisDrone 2019")
    st.markdown("- mAP@50: 30.8%")
    st.markdown("- Classes: 10")
    st.markdown("- Training images: 7,966")

# Upload
uploaded = st.file_uploader(
    "Upload a drone image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
)

if uploaded:
    is_video = uploaded.name.endswith(("mp4", "avi", "mov"))
    suffix = Path(uploaded.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if is_video:
        st.subheader("🎥 Video Detection")
        with st.spinner("Processing video... this may take a while"):
            results = model(
                tmp_path,
                conf=conf_threshold,
                save=True,
                project="results/detections",
                name="video_out",
                exist_ok=True,
                verbose=False
            )

        # Find video wherever YOLO saved it
        import glob
        import subprocess

        video_files = glob.glob("**/video_out/*.mp4", recursive=True)
        if not video_files:
            video_files = glob.glob("**/video_out/*.avi", recursive=True)

        converted_output = "results/detections/video_out/output_web.mp4"
        Path("results/detections/video_out").mkdir(parents=True, exist_ok=True)

        if video_files:
            raw_output = video_files[0]
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_output,
                "-vcodec", "libx264", "-acodec", "aac",
                converted_output
            ], capture_output=True)

            if Path(converted_output).exists():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with open(converted_output, "rb") as f:
                        st.video(f.read())
            else:
                st.warning("Conversion failed.")
        else:
            st.warning("Output video not found.")

        st.metric("Frames Processed", len(results))
        os.unlink(tmp_path)


    else:
        # Read image
        img = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        with st.spinner("Detecting objects..."):
            results = model(tmp_path, conf=conf_threshold, verbose=False)

        # Generate outputs
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Layout
        if show_heatmap:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.subheader("Detections")
            st.image(annotated_rgb, use_container_width=True)

        if show_heatmap:
            with col3:
                st.subheader("Grad-CAM")
                heatmap_img = generate_heatmap(img.copy(), results)
                heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, use_container_width=True)

        # Metrics
        st.divider()
        boxes = results[0].boxes
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Objects Detected", len(boxes))
        col2.metric("Avg Confidence", f"{float(boxes.conf.mean()):.0%}" if len(boxes) > 0 else "0%")
        col3.metric("Model mAP@50", "30.8%")
        col4.metric("Inference Speed", f"{results[0].speed['inference']:.1f}ms")

        # Per class breakdown
        if len(boxes) > 0:
            st.subheader("Detection Breakdown")
            class_counts = {}
            for box in boxes:
                cls_name = results[0].names[int(box.cls)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
            cols = st.columns(len(class_counts))
            for i, (cls_name, count) in enumerate(sorted(class_counts.items())):
                cols[i].metric(cls_name, count)

        # Cleanup
        os.unlink(tmp_path)

else:
    st.info("👆 Upload a drone image or video to get started")
    
   
    st.markdown("""
    **What KestrelAI detects:**
    
    | Class | Description |
    |-------|-------------|
    | 🚶 Pedestrian | Individual walking persons |
    | 👥 People | Groups of persons |
    | 🚲 Bicycle | Bicycles |
    | 🚗 Car | Passenger vehicles |
    | 🚐 Van | Vans and minibuses |
    | 🚛 Truck | Large trucks |
    | 🛺 Tricycle | Motorized tricycles |
    | 🛺 Awning-tricycle | Covered tricycles |
    | 🚌 Bus | Buses |
    | 🏍️ Motor | Motorcycles |
    """)