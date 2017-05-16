from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from load import loadData
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Load in raw data
data = pickle.load(open( "dataset400.dat", "rb" ))

X, y, X_test, y_test, num_classes = loadData(flatten=True, normalize=False, type='int32')
#X = X[0::3]
#y = y[0::3]
#X_test = X_test[1::2]
#y_test = y_test[1::2]

# Printing info from the loaded data
n_features = X.shape[1]
n_points = y.shape[0]
n_test = y_test.shape[0]
print("Total dataset size:")
print("n_features: %d" % n_features)
print("n_trainpoints: %d" % n_points)
print("n_testpoints: %d" % n_test)

# Carrying out PCA/Eigenfaces, dimensionality reduction for our data
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, 48, 48))


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print("n_PCAfeatures: %d" % X_train_pca.shape[1])


# Beginning SVM model fitting
print("Fitting the classifier to the training set")
t0 = time()

# Full parameter grid below 
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# Single parameter for testing
param_grid = {'C': [5],
              'gamma': [0.005], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Printing results of trained model on test data
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) # , labels=range(6)))
cv2.imwrite('eigenface0.png',eigenfaces[0].reshape((48, 48)) )
cv2.imwrite('eigenface1.png',eigenfaces[1].reshape((48, 48)) )
cv2.imwrite('eigenface2.png',eigenfaces[2].reshape((48, 48)) )
cv2.imwrite('eigenface3.png',eigenfaces[3].reshape((48, 48)) )
