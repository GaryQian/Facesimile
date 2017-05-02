import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import cPickle as pickle
from load import loadData
from imgProcessing import get400

# Load training data
'''print 'Loading training data'
imgDim = 400
data = pickle.load(open('dataset400.dat', 'rb'))
print 'Done'''

X_train, y_train, X_test, y_test, num_classes = loadData(flatten=True)

'''for arr, arr1 in X_train, X_test:
	arr = arr.flatten()
	arr1 = arr1.flatten()'''

imgDim = 48
print(X_train)
print(y_train)
print(X_train.shape)

'''#normalize inputs to 0.0-1.0 (from 0-255)
temp = np.zeros((X_train.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_train[:,:,:]
X_train = temp

#normalize inputs to 0.0-1.0 (from 0-255)
temp = np.zeros((X_test.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_test[:,:,:]
X_test = temp'''

rand = np.random.RandomState(1) 

mod = AdaBoostClassifier(n_estimators = 100, random_state = rand)


mod.fit(X_train, y_train)

total = len(y_test) 
success = 0.0
for x, y in X_test, y_test:
	pred = mod.predict(x) 
	if pred == y:
		success += 1.0

print("Success rate: " + success/total)
