import numpy as np
import cPickle as pickle
from os import path
from os import walk
from os import listdir
import os

#Import:  from load import loadData
#USAGE: X_train, y_train, X_test, y_test, num_classes = loadData()
#USAGE: X_train, y_train, X_test, y_test, num_classes = loadData(flatten=True, normalize=False, type='int32')
def loadData(flatten=False, normalize=True, type='float32'):
	print "Loading Data..."
	parts = 6

	X_train = None
	y_train = list()
	for i in range(parts):
		data = dict()
		data = pickle.load(open( "datasettrain48-" + str(i) + ".dat", "rb" ))
		if i == 0:
			X_train = np.array(data['X'])
			y_train = np.array(data['y'])
		else:
			X_train = np.vstack((X_train,data['X']))
			y_train = np.append(y_train,data['y'])
		print '\tPart ' + str(i)
	print '\tLoading Test'
	data = pickle.load(open( "datasettest48.dat", "rb" ))
	X_test = np.array(data['X'])
	y_test = np.array(data['y'])
	num_classes = max(y_train) + 1
	
	
	if flatten:
		print '\tFlattening'
		X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 
		X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

	
	X_train = np.array(X_train).astype(type)
	X_test = np.array(X_test).astype(type)
	if normalize:
	#normalize
		print '\tNormalizing'
		X_train = X_train / 255.0

		X_test = X_test / 255.0
	print 'Done'
	return (X_train, y_train, X_test, y_test, num_classes)