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


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Load in raw data
data = pickle.load(open( "dataset400.dat", "rb" ))

X, y, X_test, y_test, num_classes = loadData(flatten=True, normalize=False, type='int32')
# Used to train on a sample of the data instead of entire set
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

print("Extracting the top %d eigenfaces from %d faces" % (n_components))
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done")

eigenfaces = pca.components_.reshape((n_components, 48, 48))
print(eigenfaces[0])

# Writing out the most important eigenface
for i in range(20):
	out = eigenfaces[i]
	out = out - np.amin(out)
	out = out * (254/np.amax(out))
	cv2.imwrite('eigenface'+str(i)+'.png',out.reshape((48, 48)))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)
print("done")
print("n_PCAfeatures: %d" % X_train_pca.shape[1])


# Beginning SVM model fitting
print("Fitting the classifier to the training set")

# Full parameter grid below, used to find best params
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# Single parameter for testing, used once best params found
param_grid = {'C': [5],
              'gamma': [0.005], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y)
print("done")
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Printing results of trained model on test data
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print("done")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 

# Write out data to files
pickle.dump(pca, open( "PCA.dat", "wb" )) # PCA components
pickle.dump(clf, open( "SVM.dat", "wb" )) # SVM model
