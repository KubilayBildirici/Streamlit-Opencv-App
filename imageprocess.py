import cv2
import streamlit as st
import numpy as np
from PIL import Image,ImageEnhance,ImageOps

def CannyDetect(our_image):
    t_lower = 50
    t_upper = 150
    new_img = np.array(our_image.convert('RGB'))
    canny_rate = st.sidebar.slider("Canny", t_lower, t_upper)
    img = cv2.cvtColor(new_img, 1)
    canny = cv2.Canny(img,t_lower,t_upper,canny_rate)
    st.image(canny)



def convert_to_gray(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray)


def bluring(our_image):
    new_img = np.array(our_image.convert('RGB'))
    blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
    img = cv2.cvtColor(new_img, 1)
    blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
    st.image(blur_img)


def median_blur(our_image):
    new_img = np.array(our_image.convert('RGB'))
    blur_rate = st.sidebar.slider("median Blur", 0.5, 3.5)
    img = cv2.cvtColor(new_img, 1)
    blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
    st.image(blur_img)


def bilateral(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    bilateral = cv2.bilateralFilter(img,15,75,75)
    st.image(bilateral)


def filter2D(our_image):
    new_img = np.array(our_image.convert('RGB'))
    kernel2 = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
    img = cv2.cvtColor(new_img, 1)
    filter = cv2.filter2D(img,ddepth=-1,kernel=kernel2)
    st.image(filter)


def brightness(our_image):
    c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
    enhancer = ImageEnhance.Brightness(our_image)
    img_output = enhancer.enhance(c_rate)
    st.image(img_output)


def frontalface(img):
    face = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detect = face.detectMultiScale(img_gray,1.2,4)
    for (x,y,w,h) in face_detect:
        cv2.rectangle(img_gray,(x,y),(x+w,y+h),(255,0,0),6)
    return img_gray

def detect_eye(img):
    eye = cv2.CascadeClassifier("cascade/haarcascade_eye.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eye_detect = eye.detectMultiScale(gray,1.2,3)
    for(x,y,w,h) in eye_detect:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),6)
    return gray













