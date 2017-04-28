import cv2
import numpy as np
import cPickle as pickle
from os import path
from os import walk
from os import listdir
import os

from imgProcessing import get400


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load training data
print 'Loading Training Data'
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
print 'Pickled ' + str(count) + ' images'