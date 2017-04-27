import numpy as np
import cv2




def get400(img):

	# Pretrained classifiers for face detection
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	# Return a gray image of the given image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Return a list of rectangles of detected faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    xMid = x + w/2
	    yMid = y + h/2
	    cv2.circle(img,(xMid,yMid), 5, (255,0,0), -1)
	    cv2.rectangle(img,(xMid-200,yMid-200),(xMid+200,yMid+200),(255,0,0),2)

	# Write image, this is just for testing.
	cv2.imwrite('testOuput.png',img)

#def main():
img = cv2.imread("test.png")
get400(img)
print("hello")