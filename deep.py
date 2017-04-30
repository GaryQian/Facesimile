import cv2
import numpy as np
import cPickle as pickle
from os import path
from os import walk
from os import listdir
import os
import csv
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils

from imgProcessing import get400



# (X_train1, y_train1), (X_test1, y_test1) = cifar10.load_data()
# print(X_train1.shape)
# print(y_train1.shape)


K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load training data
print 'Loading Training Data'
imgDim = 48
#data = pickle.load(open('dataset400.dat','rb'))

'''X_test = list()
y_test = list()
X_train = list()
y_train = list()
tc = dict()
map = dict()
for i in range(7):
	tc[i] = 0
	map[i] = i - 1
map[0] = 0
with open('./fer2013/fer2013.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	count = 0
	for row in spamreader:
		if row[0] != '1':
			if row[2] == 'Training':
				X_train.append(np.reshape(row[1].split(' '), (-1, imgDim)))
				y_train.append(map[int(row[0])])
				tc[int(row[0])] += 1
			elif 'Test' in row[2]:
				X_test.append(row[1].split(' '))
				y_test.append(map[int(row[0])])
				tc[int(row[0])] += 1
			count += 1
			if count % 2500 == 0:
				print 'Loaded ' + str(count)

print 'Done'
for i in range(7):
	print str(i) + ' ' + str(tc[i])
data = dict()
data['X'] = X_train
data['y'] = y_train
pickle.dump(data, open( "datasettrain48.dat", "wb" ))
data['X'] = X_test
data['y'] = y_test
pickle.dump(data, open( "datasettest48.dat", "wb" ))'''
data = dict()
data = pickle.load(open( "datasettrain48.dat", "rb" ))
X_train = data['X']
y_train = data['y']

data = pickle.load(open( "datasettest48.dat", "rb" ))
X_test = data['X']
y_test = data['y']

num_classes = max(y_train) + 1

'''for subj in range(0,1000):
	subjstr = str(subj)
	if (subj < 10): subjstr = '00' + subjstr
	if (subj >= 10 and subj < 100): subjstr = '0' + subjstr
	subjstr = 'S' + subjstr
	for i in range(15):
		samplestr = ''
		if (i < 10): samplestr = '00' + str(i)
		else: samplestr = '0' + str(i)
		dir = rootdir + '/cohn-kanade-images' + '/' + subjstr + '/' + samplestr
		fname = dir + '/' + subjstr + '_' + samplestr + '_' + '00000' + samplestr + '.png'
		#print fname
		if path.exists(fname):
			pass
			#labelpath = fname[:-4] + '_emotion.txt
			#print labelpath'''
'''num_classes = 2
X_train = np.ndarray(shape0)
y_train = np.ndarray((imNum0))

for x in range(0, imNum1): #for each image in trainingData/0/*
	fname = prestring1 + str(x) + postString
	if path.exists(fname):
		temp = cv2.imread(fname)
		X_train[count] = temp
		y_train[count] = 1

		count += 1
	'''
print 'Preprocessing'
#X_train = X_train[:count]
#y_train = y_train[:count]
# normalize inputs from 0-255 to 0.0-1.0
X_train = np.array(X_train).astype('float32')
X_train = X_train / 255.0
temp = np.zeros((X_train.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_train[:,:,:]
X_train = temp

print X_train.shape
# one hot encode outputs
#y_train = y_train.reshape((-1, 1))
#y_train = np_utils.to_categorical(y_train)

print 'Done'

# Create the model
print 'Constructing Model'

'''
#deeper model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgDim,imgDim, 1), activation='relu', padding='same'))
#model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.2))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))'''


model = load_model('modeldeep.dat')
print 'Done'


print 'Compiling'
# Compile model
epochs = 200
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


print 'Fitting model'
# Fit the model
#for i in range(len(X_train)):
model.fit(X_train[5000:10000], y_train[5000:10000], batch_size=256, epochs=epochs, verbose=1, callbacks=[], validation_data=(X_train[:2000], y_train[:2000]), shuffle=True, class_weight=None, sample_weight=None)
#model.fit(X_train, y_train, validation_data=(X_train, y_train), nb_epoch=epochs, batch_size=32)
'''validation_split=0.2,'''
print 'Done'

print 'Saving data'
model.save('modeldeep.dat')

print 'Done'

print model.predict_proba(X_train[930:1000], batch_size=32, verbose=1)

"""
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
