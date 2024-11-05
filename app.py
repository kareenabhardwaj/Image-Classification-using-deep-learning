import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load a pre-trained model (MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

# Function to make predictions
def predict(image):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Streamlit app
st.title("Image Classification App")
st.write("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict(image)
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")
