import numpy as np
import cPickle as pickle
from os import path
from os import walk
from os import listdir
import os


#USAGE: X_train, y_train, X_test, y_test, num_classes = loadData()
def loadData():
	print "Loading Data..."
	parts = 3
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

	data = pickle.load(open( "datasettest48.dat", "rb" ))
	X_test = data['X']
	y_test = data['y']
	num_classes = max(y_train) + 1
	
	#normalize
	print '\tNormalizing'
	X_train = np.array(X_train).astype('float32')
	X_train = X_train / 255.0

	X_test = np.array(X_test).astype('float32')
	X_test = X_test / 255.0
	print 'Done'
	return (X_train, y_train, X_test, y_test, num_classes)