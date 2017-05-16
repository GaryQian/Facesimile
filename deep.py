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
from load import loadData



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

X_train, y_train, X_test, y_test, num_classes = loadData()

#num_classes = max(y_train) + 1


print 'Preprocessing'
temp = np.zeros((X_train.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_train[:,:,:]
X_train = temp

temp = np.zeros((X_test.shape[0],imgDim,imgDim,1))
temp[:,:,:,0] = X_test[:,:,:]
X_test = temp
print 'Done'

# Create the model
print 'Constructing Model'

'''#Shallowest model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgDim,imgDim, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
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

#Med model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(imgDim,imgDim, 1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
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
model.add(Dense(num_classes, activation='softmax'))

'''
#Shallow model
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

'''#Deeper model
model = Sequential()
model.add(Convolution2D(64, (3, 3), input_shape=(imgDim,imgDim, 1), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))'''

modelName = 'modelshallow.dat111'
model = load_model(modelName)
print 'Done'


print 'Compiling'
# Compile model
epochs = 50
lrate = 0.04
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#Wsave = model.get_weights()
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.set_weights(Wsave)
print(model.summary())


print 'Fitting model'
# Fit the model
#for i in range(len(X_train)):
model.fit(X_train[0:], y_train[0:], batch_size=256, epochs=epochs, verbose=1, callbacks=[], validation_data=(X_test[0:], y_test[0:]),   shuffle=True, class_weight=None, sample_weight=None)
#model.fit(X_train, y_train, validation_data=(X_train, y_train), nb_epoch=epochs, batch_size=32)
'''validation_split=0.2,'''
'''validation_data=(X_test[0:], y_test[0:]), '''
print 'Done'

print 'Saving data'
model.save(modelName + '1')

print 'Done'

print model.predict_proba(X_train[930:1000], batch_size=32, verbose=1)

"""
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
