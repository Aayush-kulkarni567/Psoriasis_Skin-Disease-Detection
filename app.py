import streamlit as st

import tensorflow as tf

import numpy as np

from PIL import Image

from keras.models import load_model

from keras.applications.efficientnet import preprocess_input


# Load model
model = load_model("psoriasis_model.h5")


# Title
st.title("Psoriasis Skin Disease Detection")


# Upload image
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)

    # Get predicted class
    predicted_class = np.argmax(prediction)

    # Display result
    if predicted_class == 0:
        st.success("Normal Skin Detected")

    else:
        st.error("Psoriasis Detected")