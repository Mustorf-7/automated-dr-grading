
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import io
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
IMG_SIZE    = 224
THRESHOLD   = 5.5
BEST_KAPPA  = 0.8364

GRADE_INFO = {
    0: {"label": "No DR",         "color": "#2ecc71", "action": "Routine screening in 12 months"},
    1: {"label": "Mild",          "color": "#f1c40f", "action": "Follow-up in 6-12 months"},
    2: {"label": "Moderate",      "color": "#e67e22", "action": "Referral to ophthalmologist recommended"},
    3: {"label": "Severe",        "color": "#e74c3c", "action": "Urgent referral required"},
    4: {"label": "Proliferative", "color": "#8e44ad", "action": "Immediate specialist intervention needed"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features   = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Mustorf/dr-grading-resnet50",
        filename="best_model.pth"
    )
    m    = DRClassifier().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

# ── Preprocessing ─────────────────────────────────────────────────────────────
def check_quality(pil_img):
    gray  = np.array(pil_img.convert("L"))
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= THRESHOLD, round(score, 2)

def apply_clahe(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))

# ── TTA ───────────────────────────────────────────────────────────────────────
def tta_predict(model, pil_img):
    tta_transforms = [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomRotation((90, 90)),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomRotation((270, 270)),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN, STD)]),
    ]
    probs_list = []
    with torch.no_grad():
        for tfm in tta_transforms:
            t = tfm(pil_img).unsqueeze(0).to(device)
            probs_list.append(torch.softmax(model(t), dim=1).cpu())
    avg_probs = torch.stack(probs_list).mean(dim=0).squeeze()
    return avg_probs.argmax().item(), avg_probs.max().item(), avg_probs.tolist()

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def run_gradcam(model, pil_img, class_idx):
    gradients, activations = [None], [None]

    def fwd_hook(m, i, o): activations[0] = o.detach()
    def bwd_hook(m, gi, go): gradients[0] = go[0].detach()

    h1 = model.backbone.layer4[-1].register_forward_hook(fwd_hook)
    h2 = model.backbone.layer4[-1].register_full_backward_hook(bwd_hook)

    tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])
    t = tfm(pil_img).unsqueeze(0).to(device)
    t.requires_grad = True

    logits = model(t)
    model.zero_grad()
    logits[0, class_idx].backward()

    weights = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam     = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam     = torch.nn.functional.interpolate(
                cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam     = cam.squeeze().cpu().numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()

    orig    = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.4 * heatmap + 0.6 * orig).astype(np.uint8)
    return cam, overlay

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DR Grading System", page_icon="👁️", layout="wide")

st.title("👁️ Automated Diabetic Retinopathy Grading")
st.markdown(
    "Upload a **retinal fundus image** to receive an automated DR severity grade "
    "with Grad-CAM explainability."
)
st.divider()

model    = load_model()
uploaded = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(pil_img, use_container_width=True)

    # Quality check
    quality_ok, quality_score = check_quality(pil_img)
    st.subheader("Pipeline Steps")
    step1, step2, step3 = st.columns(3)
    step1.metric("① Quality Score", f"{quality_score}",
                 delta="Pass" if quality_ok else "Fail")

    if not quality_ok:
        st.error(f"Image rejected: Quality score {quality_score} is below "
                 f"threshold {THRESHOLD}. Please upload a sharper fundus image.")
        st.stop()

    # CLAHE
    clahe_img = apply_clahe(pil_img)
    step2.metric("② CLAHE", "Applied", delta="Contrast enhanced")

    # TTA inference
    with st.spinner("Running TTA inference (5 augmentations)..."):
        grade, confidence, all_probs = tta_predict(model, clahe_img)

    step3.metric("③ TTA Inference", "Complete", delta="5 augmentations averaged")

    # Result
    st.divider()
    info = GRADE_INFO[grade]
    st.markdown(f"""
    <div style="background:{info["color"]}22; border-left:5px solid {info["color"]};
                padding:20px; border-radius:8px; margin:10px 0">
        <h2 style="color:{info["color"]}; margin:0">
            Grade {grade} — {info["label"]}
        </h2>
        <p style="font-size:18px; margin:8px 0">
            Confidence: <strong>{confidence:.1%}</strong>
        </p>
        <p style="font-size:16px; margin:0">
             <em>{info["action"]}</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.subheader("Grade Probability Distribution")
    prob_cols = st.columns(5)
    for i, (col, prob) in enumerate(zip(prob_cols, all_probs)):
        col.metric(f"Grade {i}", f"{prob:.1%}", delta=GRADE_INFO[i]["label"])

    # Grad-CAM
    st.subheader("Grad-CAM Explainability")
    with st.spinner("Generating Grad-CAM heatmap..."):
        cam, overlay = run_gradcam(model, clahe_img, grade)

    g1, g2, g3 = st.columns(3)
    g1.image(np.array(clahe_img), caption="CLAHE Preprocessed",
             use_container_width=True)
    g2.image((cm.jet(cam)[:, :, :3] * 255).astype(np.uint8),
             caption="Grad-CAM Heatmap (red = high attention)",
             use_container_width=True)
    g3.image(overlay, caption="Overlay", use_container_width=True)

    st.caption(
        "Red regions indicate retinal areas most influential in the model's decision. "
        "In DR, these typically correspond to microaneurysms, haemorrhages, or neovascularisation."
    )

    st.divider()
    st.caption(
        f"Model: ResNet50 | TTA: 5 augmentations | "
        f"Best Val Kappa: {BEST_KAPPA} | Dataset: APTOS 2019"
    )
