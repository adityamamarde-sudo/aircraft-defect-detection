import os
import urllib.request
import streamlit as st
import torch

MODEL_PATH = "aircraft_defect_model_3datasets.pth"
HF_MODEL_URL = "https://huggingface.co/Aditya-Mamarde/aircraft-defect-detector/resolve/main/aircraft_defect_model_3datasets.pth"


@st.cache_resource
def load_defect_detector():
    if not os.path.exists(MODEL_PATH):
        with st.spinner(
            "Downloading AI model weights from Hugging Face (166 MB)... Please wait."
        ):
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)

    device = torch.device("cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    return model
