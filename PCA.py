import numpy as np
import cPickle as pickle

data = pickle.load(open( "dataset400.dat", "rb" ))
X = data['X']
y = data['y']

print(len(X))
print(X[0].shape)
#print(y[0].shape)
for img in X:
	img = img.flatten()

print(len(X))
print(X[0].shape)
print(len(y))
print(y[0])