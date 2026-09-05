import base64
import os
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
    initial_sidebar_state="expanded",
)

MODEL_PATH = "aircraft_defect_model_3datasets.pth"
HF_MODEL_URL = "https://huggingface.co/Aditya-Mamarde/aircraft-defect-detector/resolve/main/aircraft_defect_model_3datasets.pth"
VIDEO_PATH = "intro_animation.mp4"

# Initialize Session State
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False


# -------------------------------------------------------------------
# Model Loading Logic
# -------------------------------------------------------------------
@st.cache_resource
def load_defect_detector():
    """Downloads weights from Hugging Face if missing and initializes the model."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner(
            "Downloading AI model weights from Hugging Face (166 MB)... Please wait."
        ):
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)

    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Rebuild Faster R-CNN architecture matching trained weights (6 classes)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        num_classes = 6
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


# Load model at startup
try:
    model = load_defect_detector()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -------------------------------------------------------------------
# Inference Helper Function
# -------------------------------------------------------------------
def predict_defects(image_pil, confidence_threshold=0.5):
    """Runs inference on the input image and draws detection bounding boxes."""
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
# SCREEN 1: Full-Screen Autoplay Intro Video with Auto-Transition
# -------------------------------------------------------------------
if not st.session_state["show_dashboard"]:
    if os.path.exists(VIDEO_PATH):
        # Convert video to base64 for seamless HTML rendering
        with open(VIDEO_PATH, "rb") as video_file:
            video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")

        # HTML5 Video element with JavaScript auto-click on finish
        video_html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%;">
            <video id="introVideo" width="100%" style="max-height: 80vh; border-radius: 10px;" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <script>
            var video = document.getElementById("introVideo");
            video.onended = function() {{
                // Trigger transition button click when video ends
                const buttons = window.parent.document.querySelectorAll('button');
                for (let btn of buttons) {{
                    if (btn.innerText.includes("Skip Intro") || btn.innerText.includes("Dashboard")) {{
                        btn.click();
                        break;
                    }}
                }}
            }};
        </script>
        """
        st.components.v1.html(video_html, height=550)

    # Fallback/Skip button (hidden automatically when JS triggers it on end)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "⏩ Skip Intro to Dashboard",
            type="primary",
            use_container_width=True,
        ):
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
