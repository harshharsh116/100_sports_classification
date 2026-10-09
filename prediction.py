import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import json
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
    model = tf.keras.models.load_model(extract_dir)
    return model


model = load_sports_model()

st.title("🏅 Sports Classification")
uploaded_file = st.file_uploader(
    "Upload a sports image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = image.resize((299, 299))

    # Convert to numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])

    predicted_sport = class_names[predicted_index]

