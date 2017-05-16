import numpy as np
import cv2

# Load trained SVM Model
print("Loading trained SVM Model and PCA information")


# Capture video from computer camera
cap = cv2.VideoCapture(0)


# Run program till q is pressed
while(True):

	# Capture frame-by-frame
	ret, frame = cap.read()
	# Resize frame to proper size
	
	# Project image on eigenfaces orthonormal basis

	# Use pca image to predict output
	
	# Overlay prediction on output image

	# Display the resulting frame
	cv2.imshow('Camera Alignment Assistant',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
