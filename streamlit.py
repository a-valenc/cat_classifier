import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Set expected image size (based on your model)
IMG_SIZE = (128, 128)

# Class labels
class_labels = [
    "American Shorthair", "Bengal", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian",
    "Ragdoll", "Siamese", "Sphynx", "Tuxedo"
]

# Load model once and cache it
@st.cache_resource
def load_trained_model():
    return load_model("v2_model.keras")

model = load_trained_model()

# Preprocessing function using raw image bytes
def preprocess_image(image_bytes, target_size=IMG_SIZE):
    img = load_img(BytesIO(image_bytes), target_size=target_size)
    img_array = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict(image_bytes):
    processed = preprocess_image(image_bytes)
    preds = model.predict(processed)[0]
    index = np.argmax(preds)
    confidence = preds[index] * 100  # Convert to percentage
    return index, confidence

# Streamlit UI
st.set_page_config(page_title="🐱 Cat Breed Classifier", layout="centered")
st.title("🐱 Cat Breed Classifier")

uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    image_bytes = uploaded_file.read()
    predicted_index, confidence = predict(image_bytes)

    # Show result
    st.markdown(f"### 🐾 Predicted Breed: **{class_labels[predicted_index]}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
