import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from huggingface_hub import hf_hub_download
with open("class_names.json", "r") as f:
    class_names = json.load(f)

@st.cache_resource
def load_sports_model():
    model_path = hf_hub_download(
        repo_id="harsh8611/sports-classification-model",
        filename="best_sports_model.keras",
        repo_type="model"
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

model = load_sports_model()

st.title("Sports Classification")

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



    st.success(f"Predicted sport: {predicted_sport}")











