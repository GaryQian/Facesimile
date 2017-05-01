import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import cPickle as pickle

from imgProcessing import get400

# Load training data
print 'Loading training data'
imgDim = 400
data = pickle.load(open('dataset400.dat', 'rb'))
print 'Done'
