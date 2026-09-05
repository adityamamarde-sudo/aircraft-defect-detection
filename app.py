import os
import time
import urllib.request
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Aircraft Surface Defect Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "aircraft_defect_model_3datasets.pth"
HF_MODEL_URL = "https://huggingface.co/Aditya-Mamarde/aircraft-defect-detector/resolve/main/aircraft_defect_model_3datasets.pth"
VIDEO_PATH = "intro_animation.mp4"

# Buffer (in seconds) to account for browser loading and prevent cutting off trailing audio
PLAYBACK_BUFFER = 2.0

if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False


# -------------------------------------------------------------------
# Model Loading Logic
# -------------------------------------------------------------------
@st.cache_resource
def load_defect_detector():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model weights (166 MB)..."):
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)

    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        num_classes = 6
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


try:
    model = load_defect_detector()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -------------------------------------------------------------------
# Video Duration Helper
# -------------------------------------------------------------------
def get_video_duration(file_path):
    """Calculates exact duration of the video in seconds."""
    try:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass
    return 8.0  # Fallback duration if metadata cannot be read


# -------------------------------------------------------------------
# Inference Helper Function
# -------------------------------------------------------------------
def predict_defects(image_pil, confidence_threshold=0.5):
    image_rgb = image_pil.convert("RGB")
    img_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    img_cv = np.array(image_rgb)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    detections_found = 0

    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            detections_found += 1
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f"Defect Class {label}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_cv,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                img_cv,
                label_text,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return result_img, detections_found


# -------------------------------------------------------------------
# SCREEN 1: Full-Screen Intro Video (Buffered Auto-Transition)
# -------------------------------------------------------------------
if not st.session_state["show_dashboard"]:
    if os.path.exists(VIDEO_PATH):
        st.markdown(
            """
            <style>
                header, footer, [data-testid="stSidebar"] {
                    display: none !important;
                }
                .main .block-container {
                    padding: 0 !important;
                    margin: 0 !important;
                    max-width: 100vw !important;
                    height: 100vh !important;
                    background-color: black;
                }
                video {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    object-fit: cover !important;
                    z-index: 1000 !important;
                }
                .stButton button {
                    position: fixed !important;
                    top: 20px !important;
                    right: 25px !important;
                    z-index: 10000 !important;
                    background: rgba(0, 0, 0, 0.6) !important;
                    color: white !important;
                    border: 1px solid rgba(255, 255, 255, 0.7) !important;
                    border-radius: 6px !important;
                    padding: 8px 18px !important;
                    font-family: sans-serif !important;
                }
                .stButton button:hover {
                    background: rgba(255, 255, 255, 0.2) !important;
                    border-color: white !important;
                    color: white !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Manual Skip button
        if st.button("Skip Intro ✕"):
            st.session_state["show_dashboard"] = True
            st.rerun()

        # Stream native video component directly
        st.video(VIDEO_PATH, autoplay=True)

        # Wait duration of the video + the safety buffer
        raw_duration = get_video_duration(VIDEO_PATH)
        total_wait_time = raw_duration + PLAYBACK_BUFFER
        time.sleep(total_wait_time)

        # Flip state and advance to dashboard
        st.session_state["show_dashboard"] = True
        st.rerun()
    else:
        st.session_state["show_dashboard"] = True
        st.rerun()

# -------------------------------------------------------------------
# SCREEN 2: Main Inspection Dashboard
# -------------------------------------------------------------------
else:
    st.title("✈️ Aircraft Surface Defect Detection System")
    st.write(
        "Upload an image of an aircraft surface to identify structural defects in real-time."
    )

    # Sidebar Controls
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.30,
        step=0.05,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("◀ Replay Intro Video"):
        st.session_state["show_dashboard"] = False
        st.rerun()

    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload an inspection image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Inspection Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Defect Detection Output")
            with st.spinner("Analyzing aircraft surface..."):
                result_img, count = predict_defects(
                    image, confidence_threshold=confidence_threshold
                )

            st.image(result_img, use_container_width=True)

            if count > 0:
                st.error(f"⚠️ Detections Found: {count} defect(s) identified.")
            else:
                st.success(
                    "✅ No surface defects detected above the chosen confidence threshold."
                )
