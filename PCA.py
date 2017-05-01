import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Load in raw data
data = pickle.load(open( "dataset400.dat", "rb" ))
X = data['X']
y = data['y']

# Number of data points
N = len(X)

# Flatten all images into vectors
for i in range (0,N):
	X[i] = X[i].flatten()

# Make lists into np.arrays
X = np.array(X)
y = np.array(y)
print("Data matrix size:")
print(X.shape)
print()

# Normalize the data
X = normalize(X, axis=1, norm='l2')

# Executing PCA
pca_2 = PCA(n_components=326)
X_new = pca_2.fit_transform(X)
print("Post PCA data matrix size:")
print(X_new.shape)
print()
print("Explained variance")
print(pca_2.explained_variance_)