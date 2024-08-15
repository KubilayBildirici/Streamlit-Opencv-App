import cv2
import streamlit as st
import numpy as np


def CannyDetect(img):
    edges = cv2.Canny(img,100,200)
    return edges

def convert_to_gray(img):
    convert_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return convert_gray

def bluring(img):
    bluring_img = cv2.blur(img,(1,1))
    return bluring_img


def median_blur(img):
    median = cv2.medianBlur(img,5)
    return median

def bilateral(img):
    bilateral = cv2.bilateralFilter(img,15,75,75)
    return bilateral

def filter2D(img):
    kernel2 = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
    filter = cv2.filter2D(img,ddepth=-1,kernel=kernel2)
    return filter

def frontalface(img):
    face = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_detect = face.detectMultiScale(gray,1.2,4)
    for (x,y,w,h) in face_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),6)
    return img

def detect_eye(img):
    eye = cv2.CascadeClassifier("cascade/haarcascade_eye.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eye_detect = eye.detectMultiScale(gray,1.2,3)
    for(x,y,w,h) in eye_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),6)
    return img













