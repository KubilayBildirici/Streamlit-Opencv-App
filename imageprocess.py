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



