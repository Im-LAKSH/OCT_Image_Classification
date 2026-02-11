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
    
    st.warning("⚠️ **Disclaimer:** This tool is for educational/research purposes only.")

st.title("👁️ Retinal Disease Diagnosis System")
st.write("Upload a high-quality OCT scan (JPEG/PNG) to get an instant analysis.")

uploaded_file = st.file_uploader("Drop your scan here...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if model is None:
        st.error("❌ Model file not found. Please upload 'OCT_ResNet_95_Plus.pth' to the app folder.")
    else:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.subheader("1. Your Scan")
            st.image(image, use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        if st.button('🔍 Analyze Retina Now', type="primary"):
            with st.spinner('Analyzing retinal layers...'):
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)

                class_names = ['CSR', 'Diabetic Retinopathy', 'Macular Hole', 'NORMAL']
                pred_label = class_names[predicted.item()]
                conf_score = confidence.item() * 100

                target_layers = [model.layer4[-1]]
                cam = GradCAM(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                img_resized = np.float32(image.resize((224, 224))) / 255
                heatmap = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

            with col2:
                st.subheader("2. AI Analysis")
                
                if pred_label == "NORMAL":
                    st.success(f"### Diagnosis: {pred_label}")
                    st.balloons()
                else:
                    st.error(f"### Diagnosis: {pred_label}")
                
                st.progress(int(conf_score))
                st.caption(f"Confidence Score: **{conf_score:.2f}%**")
                
                st.write("---")
                st.write("**Attention Map (Why the AI thinks so):**")
                st.image(heatmap, caption="Red areas = Disease Location", use_column_width=True)

            with st.expander("📊 See Detailed Probability Report"):
                st.write("Model Confidence per Class:")
                probs_np = probs.cpu().numpy()[0]
                for i, name in enumerate(class_names):
                    st.write(f"- **{name}**: {probs_np[i]*100:.2f}%")
