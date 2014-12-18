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
from pickle import load

#categories = [f for f in listdir('../101_ObjectCategories') if isdir(join('../101_ObjectCategories',f)) and f != 'BACKGROUND_Google']
categories = ['dalmatian', 'laptop']

f = open('color_dict_hist.pickle','r')
clusters = load(f)
f.close()

f1 = open('BOC.pickle','wt')

color_clusters = 16 # number of possible color clusters
mini_size = 16 # reduced size of the image (16x16)
images_per_cat = 50 #

cache = {}
total_images = np.zeros((images_per_cat*len(categories), color_clusters))
count = 0 # index for master array

for c_idx in range(len(categories)):
	category = categories[c_idx]
	images = listdir(join('../101_ObjectCategories',category))
	# only worry about object categories that have at least 50 images
	if len(images) < images_per_cat:
		continue
	images = images[0:images_per_cat]
	print images
	print category + " " + str(len(images))
	cache[category] = []

	for image in images:

		file_name = join(join('../101_ObjectCategories',category),image)
		im = cv2.imread(file_name)

		#shrink image, convert to hsv, flatten
		im_small = cv2.resize(im, (mini_size,mini_size))
		hsv = cv2.cvtColor(im_small, cv2.COLOR_RGB2HSV).reshape(mini_size*mini_size,3)

		# store descriptors in cache...use binary counts or actual values?
		# storing as counts allows you to downweight the values...
		# normalize counts as unit vectors for comparison?
		#cache[category].append(np.array(hsv,dtype=np.uint8))

		# by keeping the colors together in the image, you preserve the global information

		#match each color to a histogram bin and increment its count
		hist_im = np.zeros(color_clusters)
		for i in range(len(hsv)):
			color_code = clusters.predict(hsv[i])[0] # match color to closest histogram bin
			hist_im[color_code] = hist_im[color_code] + 1 # increment corresponding histogram bin 

		total_images[count] = hist_im # put image's histogram in master array
		cache[category].append(np.array(hist_im,dtype=np.uint8))
		count = count + 1


transformer = TfidfTransformer()
tfidf = transformer.fit_transform(total_images)
print transformer.idf_ 

# store new values from tfidf.toarray() in cache and use these instead

# now carry through tfidf weighted values by multiplying with cache...

dump(cache,f)
f1.close()

# #print the color dictionary

hsv_map = clusters.cluster_centers_.reshape(math.sqrt(color_clusters),math.sqrt(color_clusters), 3)
cvt_map = np.zeros((math.sqrt(color_clusters),math.sqrt(color_clusters), 3), dtype='uint8')

for row in range(len(hsv_map)):
	for col in range(len(hsv_map[0])):
		cvt_map[row][col][0] = int(hsv_map[row][col][0])
		cvt_map[row][col][1] = int(hsv_map[row][col][1])
		cvt_map[row][col][2] = int(hsv_map[row][col][2])

cvt = cv2.cvtColor(cvt_map, cv2.COLOR_HSV2RGB)
print cvt
plt.imshow(cvt, interpolation='none')
plt.show()