import cv2
import streamlit as st
import base64
import numpy as np
import tempfile


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



# Streamlit UI
st.markdown("""
    <style>
        .stApp {
        background: url("https://imageio.forbes.com/specials-images/imageserve/65127e7fdc3814c07ca67293/0x0.jpg?format=jpg&height=900&width=1600&fit=bounds");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)


st.title("Face Detection")
st.subheader("Either Open Camera And Detect Faces Or Upload An Image And Detect Faces ")



if st.sidebar.button("Body Detect on Live Camera"):
    Bodydetect()
if st.sidebar.button("Hand Detect on Live Camera"):
    Handmesh()
if st.sidebar.button("Face Detect on Live Camera"):
    Facedetect()


# File uploader for detecting faces in an uploaded image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    ## convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_image.read()),dtype=np.uint8)
    img = cv2.imdecode(file_bytes,1)
    ##display original image
    st.image(img,channels="BGR",use_column_width=True)
    if st.sidebar.button("Detect Edges"):
        canny_detect = CannyDetect(img)
        st.image(canny_detect,use_column_width=True)

    if st.sidebar.button("convert to grayscale"):
        grayscale = convert_to_gray(img)
        st.image(grayscale,use_column_width=True)

    if st.sidebar.button("apply blurring filter to image"):
        bluring = bluring(img)
        st.image(bluring,use_column_width=True)

    if st.sidebar.button("apply median blur filter to image"):
        median = median_blur(img)
        st.image(median,use_column_width=True)

    if st.sidebar.button("apply bilateral filter to image"):
        bilat = bilateral(img)
        st.image(bilat,use_column_width=True)

    if st.sidebar.button("apply 2D filter to image"):
        filter = filter2D(img)
        st.image(filter,use_column_width=True)

    if st.sidebar.button("apply frontalface detect"):
        faces = frontalface(img)
        st.image(faces,use_column_width=True)

    if st.sidebar.button("apply eye detect"):
        eyes = detect_eye(img)
        st.image(eyes,use_column_width=True)












































