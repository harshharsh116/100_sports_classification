import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)
model = tf.keras.models.load_model('best_sports_model.keras')

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











