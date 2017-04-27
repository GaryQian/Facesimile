import numpy as np
import cv2

# Pretrained classifiers for 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def get400(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)