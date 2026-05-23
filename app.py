import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --------------------a--
# Page Config
# ----------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ----------------------
# Load Model
# ----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_model.h5")

model = load_model()

# ----------------------
# Session State for Navigation
# ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------------------
# Custom CSS (for UI)
# ----------------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#4CAF50;
    }
    .subtitle {
        font-size:18px;
        text-align:center;
        color:gray;
    }
    .btn-center {
        display:flex;
        justify-content:center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# HOME PAGE
# ----------------------
if st.session_state.page == "home":

    st.markdown('<div class="main-title">🧠 Brain Tumor Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered MRI analysis using Deep Learning</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966480.png", width=200)

    st.write("")
    st.write("")

    if st.button("🚀 Get Started"):
        st.session_state.page = "predict"


# ----------------------
# PREDICTION PAGE
# ----------------------
elif st.session_state.page == "predict":

    st.title("🔍 Upload MRI Image")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert to OpenCV format
        img = np.array(image)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.reshape(img, [1,128,128,3])

        prediction = model.predict(img)[0][0]

        st.write("")

        if prediction > 0.5:
            st.error(f"⚠ Tumor Detected ({prediction*100:.2f}% confidence)")
        else:
            st.success(f"✅ No Tumor ({(1-prediction)*100:.2f}% confidence)")

    st.write("")
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"