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

#test set of categories for testing
categories = ['dalmatian', 'pedest',  'laptop', 'airplanes', 'brain', 'kangaroo', 'chair', 'buddha', 'scorpion', 'grand_piano', 'Leopards', 'Motorbikes']

#open the k-means cluster generated from color_dict.py
f = open('limited_16bins.pickle','r')
clusters = load(f)
f.close()

f1 = open('BOC.pickle','wt')

color_clusters = 16 # number of possible color clusters
mini_size = 16 # reduced size of the image (16x16)
images_per_cat = 50 # images to process per category

total_images = np.zeros((images_per_cat*len(categories), color_clusters))
count = 0 # index for master array
cache = {}

# create transformed images whose values are assigned to histogram values
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

		#match each color to a histogram bin and increment its count
		hist_im = np.zeros(color_clusters)
		for i in range(len(hsv)):
			color_code = clusters.predict(hsv[i])[0] # match color to closest histogram bin
			hist_im[color_code] = hist_im[color_code] + 1 # increment corresponding histogram bin 

		total_images[count] = hist_im # put image's histogram in master array
		count = count + 1

# weight the values according to tfidf
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(total_images)
print transformer.idf_ 

# store transformed images in cache according to category
cache = {}
count = 0
for c_idx in range(len(categories)):

	category = categories[c_idx]
	cache[category] = []
	for m in range(images_per_cat):
		cache[category].append(np.array(tfidf.toarray()[count],dtype=np.float64))
		count += 1

dump(cache,f1)
f1.close()
