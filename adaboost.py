import numpy as np
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import cPickle as pickle
from load import loadData
from imgProcessing import get400

# Load training data
'''print 'Loading training data'
imgDim = 400
data = pickle.load(open('dataset400.dat', 'rb'))
print 'Done'''

'''
Uncomment the two lines underneath this to generate test data file, then line
20 loads that file.
'''
#result = loadData(flatten=True)
#pickle.dump(result, open('BoostData', 'wb'))

X_train, y_train, X_test, y_test, num_classes = pickle.load(open('BoostData', 'rb'))
'''for arr, arr1 in X_train, X_test:
	arr = arr.flatten()
	arr1 = arr1.flatten()'''
    
# Carrying out PCA/Eigenfaces, dimensionality reduction for our data
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, 48, 48))


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print("n_PCAfeatures: %d" % X_train_pca.shape[1])

imgDim = 48
print(X_train)
print(y_train)
print(X_train.shape)
print("X test")
print(X_test)
print("Y test")
print(y_test)

'''#normalize inputs to 0.0-1.0 (from 0-255)
temp = np.zeros((X_train.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_train[:,:,:]
X_train = temp

#normalize inputs to 0.0-1.0 (from 0-255)
temp = np.zeros((X_test.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_test[:,:,:]
X_test = temp'''


rand = np.random.RandomState(1) 
# BELOW ARE FOR ADABOOST WITH DECISION TREES
# 100 estimators --> 35% accuracy
# 150 estimators --> 36% accuracy
# 300 estimators --> 36.34% accuracy
# 450 estimators --> 36.96 % accuracy
# 600 estimators --> 36.7% accuracy

# BELOW ARE FOR AFABOOST WITH LOGISTIC REGRESSION
# 50 estimators  --> 37.4% accuracy
# 100 estimators --> 37.8% accuracy; took extremely long to run

logistic = LogisticRegression()
tree = DecisionTreeClassifier(max_depth = 10)
mod = AdaBoostClassifier(n_estimators = 500, random_state = rand)

mod.fit(X_train, y_train)
total = len(y_test)
success = 0.0
for x, y in zip(X_test, y_test):
	pred = mod.predict(x.reshape(1, -1))
	if pred == y:
		success += 1.0

print("Success rate: " + str(success/total))