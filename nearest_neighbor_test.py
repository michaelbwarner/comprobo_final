from pickle import load
import sklearn
import numpy as np
from random import sample
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn import neighbors, datasets

k = 10
color_clusters = 16
images_per_cat = 50

f = open('BOC.pickle','r')
ims = load(f)
f.close()
categories = ims.keys()
# ['dalmatian', 'pedest',  'laptop', 'airplanes', 'brain', 'kangaroo', 'chair', 'buddha', 'scorpion', 'grand_piano', 'Leopards']
target_cat = 'pedest'
# print categories

X = np.zeros((images_per_cat*len(categories), color_clusters))
y = np.zeros((images_per_cat*len(categories),))

count = 0
for im in ims[target_cat]:
	X[count] = im
	y[count] = 1
	count += 1

# populate X and y arrays from the cache
for c_idx in range(len(categories)):

	if categories[c_idx] != target_cat:
		for im in ims[categories[c_idx]]:
			X[count] = im
			y[count] = 0
			count += 1

# break data into categories for training and testing
avg_sum = 0
n_neighbors = 5
train_tests = 5
skf = StratifiedKFold(y,train_tests)
for train, test in skf:
	# print 'train', train
	# print 'test', test
	clf = neighbors.KNeighborsClassifier(n_neighbors)
	clf.fit(X[train,:],y[train])
	print clf.score(X[test,:],y[test])
	avg_sum += clf.score(X[test,:],y[test])

print 'average', avg_sum / train_tests
