import cv2
import numpy as np
import cPickle as pickle
from os import path
from os import walk
from os import listdir
import os
import csv

from imgProcessing import get400


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

'''# Load training data
print 'Loading Cohn-Kanade'
#rootdir = 'C:/Users/gary/Documents/1DOCUMENTS/Junior/ML/Facesimile'
rootdir = './'
X = []
y = []
count = 0
for subdir, dirs, files in os.walk(rootdir + '/cohn-kanade-images'):
	#print "3"
	for file in files:
		fname = path.join(subdir, file).replace('\\','/')
		labelpath = fname[:-4] + '_emotion.txt'
		labelpath = labelpath.replace('/cohn-kanade-images', '/Emotion')
		#print labelpath
		if '.png' in fname:
			
			label = -1
			try:
				f = open(labelpath, 'r')
				line = f.read()
				label = int(line[3]) - 1
			except:
				pass
			if label >= 0:
				img = cv2.imread(fname)
				img = get400(img)
				if (img is not None):
					count += 1
					X.append(img)
					y.append(label)
					cv2.imwrite(rootdir + '/dataset/img' + str(count) + '.png', img)
					if (count % 20 == 0): print 'Processed img ' + str(count)
data = dict()
data['X'] = X
data['y'] = y
pickle.dump(data, open( "dataset400.dat", "wb" ))
print 'Done'
'''


##################################################################
##################################################################
##################################################################
##################################################################


print 'Loading FER2013'
imgDim = 48
X_test = list()
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
				X_test.append(np.reshape(row[1].split(' '), (-1, imgDim)))
				y_test.append(map[int(row[0])])
				tc[int(row[0])] += 1
			count += 1
			if count % 2500 == 0:
				print 'Loaded ' + str(count)

print 'Done'
#for i in range(7):
	#print str(i) + ' ' + str(tc[i])
parts = 3
for i in range(parts):
	data = dict()
	data['X'] = X_train[int(i*len(X_train)/parts):int((i+1)*len(X_train)/parts)]
	data['y'] = y_train[int(i*len(X_train)/parts):int((i+1)*len(X_train)/parts)]
	pickle.dump(data, open( "datasettrain48-" + str(i) + ".dat", "wb" ))
	
data['X'] = X_test
data['y'] = y_test
pickle.dump(data, open( "datasettest48.dat", "wb" ))