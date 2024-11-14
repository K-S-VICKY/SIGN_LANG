import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("sign_language_model.h5")

# Define the class labels (A-Z)
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Set up Streamlit interface
st.title("Sign Language Recognition")
st.write("Upload an image of a hand gesture to recognize the corresponding sign language letter (A-Z).")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    img = img.resize((64, 64))  # Resize to match the model input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest prediction
    predicted_label = class_labels[predicted_class]

    # Display prediction
    st.title(f"Predicted Sign Language Letter: {predicted_label}")
