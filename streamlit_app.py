import base64
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)
MODEL_PATH = "waste_classifier.h5"
BACKGROUND_PATH = "background.jpg"

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")


def set_background(path):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.55), rgba(255,255,255,0.55)),
                              url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(255,255,255,0.96);
            padding: 2rem;
            border-radius: 16px;
        }}
            header[data-testid="stHeader"] {{
                background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background(BACKGROUND_PATH)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


st.title("♻️ Smart Waste Classifier")
st.write(
    "Upload a photo of a waste item and the CNN model will predict its category: "
    "cardboard, glass, metal, paper, plastic or trash."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=320)

    with st.spinner("Classifying..."):
        with tf.device("/CPU:0"):
            preds = model(preprocess(image), training=False).numpy()[0]

    top_idx = int(np.argmax(preds))
    st.success(f"Predicted: **{CLASS_NAMES[top_idx].title()}** ({preds[top_idx] * 100:.1f}% confidence)")

    st.subheader("All predictions")
    order = np.argsort(preds)[::-1]
    for i in order:
        st.progress(float(preds[i]), text=f"{CLASS_NAMES[i].title()}: {preds[i] * 100:.1f}%")
