# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from PIL import Image
import numpy as np
from ultralytics import YOLO

LOGGER = get_logger(__name__)

# Function to perform object detection with YOLO model
def detect_objects(image, column):
    # Your code for object detection with YOLOv8 goes here
    # Load in the model
    model = YOLO("best.pt")

    # Perform object detection
    prediction = model.predict(image)

    # Visualize the results
    for i, r in enumerate(prediction):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show predicted image next to the input image
    column.image(im_rgb, caption=f"Predicted Image", use_column_width=True)

def run():

    # st.write("# YOLO Welcome to Streamlit! 👋")
    st.markdown("<h1 style='text-align: center;'>Trash Detector</h1>", unsafe_allow_html=True)
    
    # Upload image
    uploaded_image = st.file_uploader("Please upload an image", type=["jpg", "jpeg", "png"])
    
    # Set up columns for predicted and original images
    col1, col2 = st.columns(2)

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        col1.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform object detection when button is clicked
        if st.button("Detect Objects"):
            # Convert image to numpy array
            image_np = np.array(image)

            # Perform object detection
            detect_objects(image_np, col2)


if __name__ == "__main__":
    run()
