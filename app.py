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
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "aircraft_defect_model_3datasets.pth"
HF_MODEL_URL = "https://huggingface.co/Aditya-Mamarde/aircraft-defect-detector/resolve/main/aircraft_defect_model_3datasets.pth"
VIDEO_PATH = "intro_animation.mp4"

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
# SCREEN 1: Event-Driven Full-Screen Video Intro
# -------------------------------------------------------------------
if not st.session_state["show_dashboard"]:
    if os.path.exists(VIDEO_PATH):
        with open(VIDEO_PATH, "rb") as f:
            video_bytes = f.read()
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")

        # Native bridge button that Streamlit listens to
        if st.button("TRANSITION_SIGNAL_BUTTON", key="transition_trigger_btn"):
            st.session_state["show_dashboard"] = True
            st.rerun()

        st.markdown(
            f"""
            <style>
                header, footer, [data-testid="stSidebar"] {{
                    display: none !important;
                }}
                .main .block-container {{
                    padding: 0 !important;
                    margin: 0 !important;
                    max-width: 100vw !important;
                    height: 100vh !important;
                    background-color: black;
                }}
                /* Hide the native Streamlit button visually */
                button[data-testid="stBaseButton-secondary"]:has(div:contains("TRANSITION_SIGNAL_BUTTON")),
                div:has(> button[key="transition_trigger_btn"]) {{
                    display: none !important;
                }}
                div[data-testid="stButton"] {{
                    position: fixed;
                    top: -100px;
                    left: -100px;
                    opacity: 0;
                    pointer-events: none;
                }}
                .video-fullscreen-wrap {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background-color: black;
                    z-index: 999998;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                video {{
                    width: 100vw;
                    height: 100vh;
                    object-fit: cover;
                }}
                .skip-btn {{
                    position: fixed;
                    top: 20px;
                    right: 25px;
                    z-index: 999999;
                    background: rgba(0, 0, 0, 0.6);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.7);
                    border-radius: 6px;
                    padding: 8px 18px;
                    font-family: sans-serif;
                    font-size: 14px;
                    cursor: pointer;
                    backdrop-filter: blur(4px);
                }}
                .skip-btn:hover {{
                    background: rgba(255, 255, 255, 0.2);
                }}
            </style>

            <button class="skip-btn" onclick="finishIntro()">Skip Intro ✕</button>

            <div class="video-fullscreen-wrap">
                <video id="introVid" autoplay playsinline>
                    <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                </video>
            </div>

            <script>
                const introVideo = document.getElementById("introVid");

                function finishIntro() {{
                    // Search document and parent frames for the trigger button
                    const doc = window.parent ? window.parent.document : document;
                    const allButtons = doc.querySelectorAll('button');
                    for (let btn of allButtons) {{
                        if (btn.innerText.includes("TRANSITION_SIGNAL_BUTTON")) {{
                            btn.click();
                            return;
                        }}
                    }}
                }}

                // Fires at the exact end of playback (audio + video)
                introVideo.onended = function() {{
                    finishIntro();
                }};
            </script>
            """,
            unsafe_allow_html=True,
        )
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
