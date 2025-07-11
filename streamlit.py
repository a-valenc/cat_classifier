import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = (96, 96)

@st.cache_resource
def load_trained_model():
    model = load_model("your_model.h5")
    return model

model = load_trained_model()

class_labels = [
    "American Shorthair", "Bengal", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian",
    "Ragdoll", "Siamese", "Sphynx", "Tuxedo"
]

def preprocess_image(image, target_size=IMG_SIZE):
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    index = np.argmax(preds)
    confidence = preds[index]
    return index, confidence, preds

st.set_page_config(page_title="🐱 Cat Breed Classifier", layout="centered")
st.title("🐱 Cat Breed Classifier")

uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    index, confidence, probs = predict(image)

    st.markdown(f"### 🐾 Predicted Breed: **{class_labels[index]}**")
    st.markdown(f"Confidence Score: `{confidence:.2f}`")

    st.markdown("### 🔍 Top 3 Predictions")
    top_indices = np.argsort(probs)[::-1][:3]
    for i in top_indices:
        st.write(f"{class_labels[i]}: {probs[i]*100:.2f}%")
