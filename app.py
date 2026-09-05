import os
import base64
import torch
import torchvision
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aircraft Surface Defect Detection", 
    page_icon="✈️", 
    layout="wide", 
    initial_sidebar_state="expanded" if st.session_state.get("intro_done", False) else "collapsed"
)

# Initialize Session State Flag
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# -----------------------------------------------------------------------------
# STEP 1: Full-Screen Splash Intro Sequence
# -----------------------------------------------------------------------------
if not st.session_state.intro_done:
    video_path = os.path.join(os.path.dirname(__file__), "intro_animation.mp4")

    # Native Python trigger button (Hidden behind iframe, auto-clicked by JS)
    if st.button("Complete Intro", key="auto_finish_btn"):
        st.session_state.intro_done = True
        st.rerun()

    if os.path.exists(video_path):
        with open(video_path, "rb") as vf:
            video_b64 = base64.b64encode(vf.read()).decode()

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body, html {{ width: 100%; height: 100%; background-color: #000; overflow: hidden; font-family: sans-serif; }}
                #container {{
                    position: fixed;
                    top: 0; left: 0; width: 100vw; height: 100vh;
                    display: flex; justify-content: center; align-items: center;
                    background-color: #000; z-index: 99999;
                }}
                video {{
                    width: 100vw; height: 100vh; object-fit: cover; display: none;
                }}
                .btn {{
                    padding: 20px 44px; font-size: 24px; font-weight: bold; color: #fff;
                    background: linear-gradient(135deg, #0072ff, #00c6ff);
                    border: none; border-radius: 50px; cursor: pointer;
                    box-shadow: 0px 4px 25px rgba(0, 198, 255, 0.6);
                    transition: transform 0.2s; z-index: 100000;
                }}
                .btn:hover {{ transform: scale(1.05); }}
            </style>
        </head>
        <body>
            <div id="container">
                <button class="btn" id="startBtn" onclick="startExperience()">▶ Launch System Experience</button>
                <video id="introVid" playsinline>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
            </div>

            <script>
                function triggerStreamlitFinish() {{
                    // Target the parent Streamlit button and click it programmatically
                    var parentDoc = window.parent.document;
                    var btn = parentDoc.querySelector('button[aria-label="Complete Intro"]');
                    if (btn) {{
                        btn.click();
                    }} else {{
                        // Fallback click search
                        var buttons = parentDoc.querySelectorAll('button');
                        for (var i = 0; i < buttons.length; i++) {{
                            if (buttons[i].innerText.includes('Complete Intro')) {{
                                buttons[i].click();
                                break;
                            }}
                        }}
                    }}
                }}

                function startExperience() {{
                    var btn = document.getElementById('startBtn');
                    var vid = document.getElementById('introVid');
                    
                    btn.style.display = 'none';
                    vid.style.display = 'block';
                    
                    vid.play().then(() => {{
                        vid.onended = function() {{
                            triggerStreamlitFinish();
                        }};
                        // Fallback auto-trigger at 9.5 seconds
                        setTimeout(triggerStreamlitFinish, 9500);
                    }}).catch(err => {{
                        console.error('Playback error:', err);
                        triggerStreamlitFinish();
                    }});
                }}
            </script>
        </body>
        </html>
        """

        st.markdown("""
            <style>
                [data-testid="stHeader"] {display: none !important;}
                .main .block-container { padding: 0 !important; max-width: 100vw !important; }
                
                /* Hide native button, keep accessible in DOM */
                div[data-testid="stButton"] {
                    position: absolute !important;
                    top: -9999px !important;
                    left: -9999px !important;
                    opacity: 0 !important;
                }
                
                iframe { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; border: none; z-index: 9999; }
            </style>
        """, unsafe_allow_html=True)

        components.html(html_code, height=1000)
        st.stop()
    else:
        st.session_state.intro_done = True
        st.rerun()

# -----------------------------------------------------------------------------
# STEP 2: Main Application Interface (Dashboard)
# -----------------------------------------------------------------------------
st.title("✈️ Aircraft Surface Defect Detection System")
st.write("Upload an image of an aircraft surface to identify structural defects in real-time.")

# Sidebar Controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.55, 
    step=0.05
)

if st.sidebar.button("Replay Intro Video"):
    st.session_state.intro_done = False
    st.rerun()

# Device & Model Setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aircraft_defect_model_3datasets.pth")
NUM_CLASSES = 6

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# File Uploader & Processing
uploaded_file = st.file_uploader("Upload an inspection image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Inspection Image")
        st.image(image, use_container_width=True)

    transform = T.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image structure..."):
        with torch.no_grad():
            predictions = model(img_tensor)

    boxes = predictions[0]['boxes'].cpu()
    scores = predictions[0]['scores'].cpu()
    labels = predictions[0]['labels'].cpu()

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    detected_count = 0
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 10, 
                f"Class {label.item()} ({score.item()*100:.1f}%)", 
                bbox=dict(facecolor='red', alpha=0.7), 
                color='white', 
                weight='bold', 
                fontsize=12
            )
            detected_count += 1

    plt.axis("off")

    with col2:
        st.subheader("AI Detection Output")
        st.pyplot(fig)
        plt.close(fig)
        
        if detected_count > 0:
            st.error(f"Detected **{detected_count}** defect(s) above {int(confidence_threshold*100)}% confidence threshold.")
        else:
            st.success("No defects detected above the threshold.")