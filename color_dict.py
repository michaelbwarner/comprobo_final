import cv2
import numpy as np
from os import listdir
from os.path import isdir, join
from pickle import dump
import sklearn.neighbors as neighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import TfidfTransformer

# this file builds a color dictionary based off the caltech library
# hsv colors are k-means grouped into 32 colors

#categories = [f for f in listdir('../101_ObjectCategories') if isdir(join('../101_ObjectCategories',f)) and f != 'BACKGROUND_Google']
categories = ['dalmatian', 'peeps',  'laptop', 'airplanes', 'brain', 'kangaroo', 'chair', 'buddha', 'scorpion', 'grand_piano', 'Leopards']
f1 = open('limited_16bins.pickle','wt')

color_clusters = 16
mini_size = 16
images_per_cat = 50

colors = np.zeros(((len(categories)-1)*images_per_cat*2*(mini_size*mini_size),3)) 
print 'total number of images', (len(categories)-1)*images_per_cat*2
cache = {}

for c_idx in range(len(categories)):
	category = categories[c_idx]
	images = listdir(join('../101_ObjectCategories',category))
	#only worry about object categories that have at least 50 images
	if len(images) < images_per_cat:
		continue
	images = images[0:images_per_cat]
	print images
	print category + " " + str(len(images))
	cache[category] = []
	pos = 0
	for image in images:
		file_name = join(join('../101_ObjectCategories',category),image)
		im = cv2.imread(file_name)

		#shrink image to 16x16, then flatten and store in a massive array
		im_small = cv2.resize(im, (mini_size, mini_size))
		hsv = cv2.cvtColor(im_small, cv2.COLOR_RGB2HSV).reshape(256,3)
		#hsv = im_small.reshape(256,3)

		#add colors to giant array
		colors[pos: pos+(mini_size * mini_size)] = hsv
		pos = pos + (mini_size * mini_size)


clusters = KMeans(n_clusters=color_clusters)
clusters.fit(colors) # fit all sampled descriptors to k means
dump(clusters,f1)
f1.close()

hsv_map = clusters.cluster_centers_.reshape(math.sqrt(color_clusters),math.sqrt(color_clusters), 3)
cvt_map = np.zeros((math.sqrt(color_clusters),math.sqrt(color_clusters), 3), dtype='uint8')

for row in range(len(hsv_map)):
	for col in range(len(hsv_map[0])):
		cvt_map[row][col][0] = int(hsv_map[row][col][0])
		cvt_map[row][col][1] = int(hsv_map[row][col][1])
		cvt_map[row][col][2] = int(hsv_map[row][col][2])

cvt = cv2.cvtColor(cvt_map, cv2.COLOR_HSV2RGB)
print cvt
print clusters.labels_
print clusters.labels_.shape
plt.imshow(cvt, interpolation='none')
plt.show()