import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile

model = tf.keras.models.load_model("/content/vehicle_Classification_Model.h5")
st.title('Vehicle Classification System')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

map_dict = {
    0: "SUV",
    1: "bus",
    2: "heavy truck",
    3: "minibus",
    4: "truck",
    5: "family sedan",
    6: "taxi",
    7: "jeep",
    8: "racing car",
    9: "fire engine",
}

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_img, (100, 100))

    st.image(opencv_img, channels="RGB")
    resized = mobile(resized)
    img_reshape = resized[np.newaxis, ...]
    generate_pred = st.button("Generate Prediction")
    if generate_pred:
        pred = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the Image is {}".format(map_dict[pred]))
