import numpy as np
import cv2
import cPickle as pickle

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imgProcessing import get400

# Load trained SVM Model
print("Loading trained SVM Model and PCA information...")
PCA = pickle.load(open( "PCA.dat", "rb" ))
SVM = pickle.load(open( "SVM.dat", "rb" ))
print("Complete")

# Capture video from computer camera
#cap = cv2.VideoCapture(0)


# Run program till q is pressed
#while(True):
if(True):

	# Capture frame-by-frame
	#ret, frame = cap.read()
	frame = cv2.imread("test.png")

	# Locate the face in the image
	face = get400(frame)

	# If a face is found
	if (face != None):
		# Resize frame to proper size
		res = cv2.resize(img,(48, 48), interpolation = cv2.INTER_CUBIC)
		res = res.reshape(1, res[0]*res[1])

		# Project image on eigenfaces orthonormal basis
		X = PCA.transform(res)

		# Use post pca image to predict output using svm
		pred = SVM.predict(X)
	
		# Overlay prediction on output image
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,pred,(10,10), font, 1,(255,255,255),2)

	# Display the resulting frame
	cv2.imwrite("sampleOutput.png",frame)
	#cv2.imshow('Camera Alignment Assistant',frame)
	#if cv2.waitKey(1) & 0xFF == ord('q'):
		#break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
