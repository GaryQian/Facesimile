Manyu Sharma - msharm27
Gary Qian - gqian1

Facesimile

Below is a brief description of the various scripts used in this project.

preprocess.py:
Function to load all training/test data from a csv file into 5 pickle output
files.

load.py:
A function written to load in training/test data from 5 pickle output files.
This function was written to take in to different parameters

imgProcessing.py:
Function which makes use of haar-features to pull the area around a face in an
image into a usable matrix. Used in live application.

SVM.py:
Function to compute eigenfaces of data and fit an SVM to the projected data
matrix. This function saves out an SVM model and PCA components.

adaboost.py:
Function to implement adaboost algorithms.

deep.py:
Function to implement deep CNNs, saves out trained model as well.

deeppca.py:
Function to implement PCA neural network, saves out trained model.

facesimile.py:
This is the real time application of our SVM. This script takes in webcam feed
of a given size (size of webcam specified in imgProcessing.py) and looks for 
faces in live feed using imgProcessing.py. From here, it processes image to 
usable format and outputs the prediction from the trained model after
projecting onto the eigenspaces, also found and saved previously. Two windows
are opened after starting this script, one which displays the webcam feed 
(larger), and one which displays the detected face image (smaller window). The
predicted sentiment is output on the top left of the larger window.