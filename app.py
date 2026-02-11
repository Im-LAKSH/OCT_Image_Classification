import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(
    page_title="RetinaAI - OCT Diagnosis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    model_path = "OCT_ResNet_95_Plus.pth"
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("RetinaAI 🩺")
    st.info("This AI tool assists in diagnosing retinal diseases from OCT scans.")
    
    st.write("### Supported Diseases:")
    st.markdown("- **CSR** (Central Serous Retinopathy)")
    st.markdown("- **Diabetic Retinopathy**")
    st.markdown("- **Macular Hole**")
    st.markdown("- **Normal Retina**")
    
    st.warning("⚠️ **Disclaimer:** This tool is for educational/research purposes only. It is not a substitute for professional medical advice.")

st.title("👁️ Retinal Disease Diagnosis System")
st.write("Upload a high-quality OCT scan (JPEG/PNG) to get an instant analysis.")

uploaded_file = st.file_uploader("Drop your scan here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if model is None:
        st.error("❌ Model file not found. Please upload 'OCT_ResNet_95_Plus.pth' to the app folder.")
    else: