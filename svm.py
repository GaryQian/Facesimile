from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from load import loadData


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

# Printing info from the loaded data
n_features = X.shape[1]
n_points = y.shape[0]
print("Total dataset size:")
print("n_features: %d" % n_features)
print("n_datapoints: %d" % n_points)

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
