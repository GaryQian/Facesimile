import numpy as np
import cv2

# Takes in an image, really a numpy array, and returns a 400x400
# image, again an array, of the area centered around a detected face
def get400(img):

	# Pretrained classifiers for face detection
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	# Return a gray image of the given image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Return a list of rectangles of detected faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	crop_img = None
	for (x,y,w,h) in faces:
	    
	    xMid = x + w/2
	    yMid = y + h/2
	    
	    # Images are 640 x 490
	    if (xMid + 200 > 640):
	    	xMid = 440
	    elif (xMid - 200 < 0):
	    	xMid = 200
	    if (yMid + 200 > 490):
	    	yMid = 290
	    elif (yMid - 200 < 0):
	    	yMid = 200

	    crop_img = gray[yMid-200:yMid+200, xMid-200:xMid+200]

	# Write image, this is just for testing.
	return crop_img

# Used for facesimile processing
def getBox(img,size):

	# Pretrained classifiers for face detection
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	# Return a gray image of the given image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Return a list of rectangles of detected faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	crop_img = None
	for (x,y,w,h) in faces:
	    
	    xMid = x + w/2
	    yMid = y + h/2
	    
	    # Images are 640 x 480
	    if (xMid + size/2 > 640):
	    	xMid = 640 - size/2
	    elif (xMid - size/2 < 0):
	    	xMid = size/2
	    if (yMid + size/2 > 480):
	    	yMid = 480 - size/2
	    elif (yMid - size/2 < 0):
	    	yMid = size/2

	    crop_img = gray[yMid-size/2:yMid+size/2, xMid-size/2:xMid+size/2]

	# Write image, this is just for testing.
	return crop_img
