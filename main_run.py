import cv2
import streamlit as st
import base64
import numpy as np
import tempfile
import time
from PIL import Image, ImageEnhance



from help import Bodydetect
from help import Handmesh
from help import Facedetect

from imageprocess import CannyDetect
from imageprocess import convert_to_gray
from imageprocess import bluring
from imageprocess import median_blur
from imageprocess import bilateral
from imageprocess import filter2D
from imageprocess import frontalface
from imageprocess import detect_eye
from imageprocess import brightness


# Streamlit UI
st.markdown("""
    <style>
        .stApp {
        background: url("https://imageio.forbes.com/specials-images/imageserve/65127e7fdc3814c07ca67293/0x0.jpg?format=jpg&height=900&width=1600&fit=bounds");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)


st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: black;
}
</style>
"""
)




st.sidebar.title(":white[_All in One with OpenCV_ ]")
with st.sidebar.chat_message("USER"):
    st.write("Hello 👋")
    st.write("My name is Kubilay")
    st.write("Welcome to my Project")
st.sidebar.image("487px-OpenCV_Logo_with_text-2.png",use_column_width=True)

activities = ["Live Camera", "Image Process","Face detect on Image","about"]
choice = st.sidebar.selectbox("Options", activities)

if choice == "Live Camera":
    if st.sidebar.button("Body Detect on Live Camera"):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success("Done!")
        Bodydetect()
    if st.sidebar.button("Hand Detect on Live Camera"):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success("Done!")
        Handmesh()
    if st.sidebar.button("Face Detect on Live Camera"):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success("Done!")
        Facedetect()


if choice == "Image Process":
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])


    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        # st.write(type(our_image))
        st.image(our_image)
        with st.status("Proccessing data..."):
            time.sleep(2)

        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Canny Detect", "bluring", "median_blur","bilateral","filter2D","brightness"])
        if enhance_type == 'Gray-Scale':
            convert_to_gray(our_image)

        elif enhance_type == 'Canny Detect':
            CannyDetect(our_image)


        elif enhance_type == 'brightness':
            brightness(our_image)


        elif enhance_type == 'bluring':
            bluring(our_image)

        elif enhance_type == "median_blur":
            median_blur(our_image)

        elif enhance_type == "bilateral":
            bilateral(our_image)

        elif enhance_type == "filter2D":
            filter2D(our_image)

        elif enhance_type == 'Original':

            st.image(our_image, width=300)
        else:
            st.subheader("The selected action could not be performed. Please try again or let us know the problem.")
            st.image(our_image, width=300)

if choice == "Face detect on Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        ## convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)


        task = ["Face detect with haarcascade","eye detect with haarcascade"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if feature_choice == "Face detect with haarcascade":
            faces = frontalface(img)
            st.image(faces, use_column_width=True)
        if feature_choice == "eye detect with haarcascade":
            eye = detect_eye(img)
            st.image(eye,use_column_width=True)

if choice == "about":
    st.balloons()
    with st.chat_message("user"):
        st.write("Hello Everyone👋")
        st.write("My name is Kubilay")
        st.write("welcome to my project")
        st.write("🌟 I hope you had fun here and liked my project 🌟")
        st.write("The project is in demo version")
        st.write("🚀 I continue to close bugs and improve the project 🚀")
        st.write("✅ You can fork the project on GitHub and help me improve it ✅")
        st.write("You can contact me via e-mail for errors and new ideas on the project.")
        st.write("kubilay_1453_25@hotmail.com")





