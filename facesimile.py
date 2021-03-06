import numpy as np
import cv2
import cPickle as pickle
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imgProcessing import getBox

# Load trained SVM Model
print("Loading trained SVM Model and PCA information...")
PCA = pickle.load(open( "PCA.dat", "rb" ))
SVM = pickle.load(open( "SVM.dat", "rb" ))
print("Complete")

# Capture video from computer camera
cap = cv2.VideoCapture(0)

# Define facial expressions
name = dict()
name['0'] = 'Angery'
name['1'] = 'Fear'
name['2'] = 'Happy'
name['3'] = 'Sad'
name['4'] = ':o'
name['5'] = ':|'

# Run program till q is pressed
t = time()
while(True):

	# Capture frame-by-frame
	ret, frame = cap.read()

	# Locate the face in the image
	face = getBox(frame, 150)

	# If a face is found
	if (face != None):
		res = cv2.resize(face,(48, 48), interpolation = cv2.INTER_CUBIC)
		res = res.reshape(1, 48*48)

		# Project image on eigenfaces orthonormal basis
		X = PCA.transform(res)

		# Use post pca image to predict output using svm
		pred = SVM.predict(X)

		# Overlay prediction on output image
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.imshow('Face Frame',face)
		cv2.putText(frame,name[str(pred[0])],(40,40), font, 1,(255,0,0),2)
	print time() - t
	t = time()
	# Display the resulting frame
	cv2.imshow('Unregistered Hypercam 3',frame)

	# Break loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
