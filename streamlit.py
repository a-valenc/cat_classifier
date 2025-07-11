import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Expected image input size
IMG_SIZE = (128, 128)

# ✅ Your actual dataset classes, in alphabetical order
class_labels = [
    "bengal", "egyptian mau", "maine coon", "persian",
    "ragdoll", "siamese", "sphynx", "tuxedo"
]

# Load the trained model only once
@st.cache_resource
def load_trained_model():
    return load_model("v2_model.keras")

model = load_trained_model()

# Preprocessing function for uploaded image
def preprocess_image(image_bytes):
    img = load_img(BytesIO(image_bytes), target_size=IMG_SIZE)
    img_array = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Run prediction
def predict(image_bytes):
    processed = preprocess_image(image_bytes)
    preds = model.predict(processed)[0]
    index = np.argmax(preds)
    confidence = preds[index] * 100
    return index, confidence

# Streamlit interface
st.set_page_config(page_title="🐱 Cat Breed Classifier", layout="centered")
st.title("🐱 Cat Breed Classifier")

uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(BytesIO(image_bytes), caption="Uploaded Image", use_container_width=True)

    index, confidence = predict(image_bytes)

    st.markdown(f"### 🐾 Predicted Breed: **{class_labels[index].title()}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
