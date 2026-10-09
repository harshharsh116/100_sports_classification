import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import json
import os
from huggingface_hub import hf_hub_download

with open("class_names.json", "r") as f:
    class_names = json.load(f)


@st.cache_resource
def load_sports_model():
    zip_path = hf_hub_download(
        repo_id="harsh8611/sports-classification-model",
        filename="best_sports_model.keras.zip",
        repo_type="model"
    )

    extract_dir = "model_extracted"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # The extracted folder contains config.json + model.weights.h5 + metadata.json
    # which together ARE the .keras model — load the folder directly
    model = tf.keras.models.load_model(extract_dir)
    return model

# ===== Model loading happens HERE, at the top level, before anything uses it =====
model = load_sports_model()
st.write("Model type:", type(model))

st.title("Sports Classification")

uploaded_file = st.file_uploader(
    "Upload a sports image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    IMG_SIZE = 299
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if not hasattr(model, "predict"):
        st.error("Model is not loaded correctly.")
        st.stop()

    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_sport = class_names[predicted_index]

    st.success(f"**Predicted Sport:** {predicted_sport}")
    st.write(f"Confidence: **{confidence*100:.1f}%**")
