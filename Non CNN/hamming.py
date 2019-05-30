import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import os
import time
from hashlib import md5
from PIL import Image, ImageFilter, ImageDraw
import scipy
import glob
from scipy import spatial

def hamming_distance(image, image2):
	score = spatial.distance.hamming(image, image2)
	return score

images = []
names = []

for img_path in glob.iglob('/Users/praky/Desktop/Work/SigVerification/*.jpeg'):
	names.append(img_path)
	# print(img_path)
	img = Image.open(img_path)
	img = img.resize((512, 256))
	# plt.imshow(img)
	# plt.plot()
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
	images.append(img)
images = np.array(images, dtype='float') / 131072.0
print(len(images))
final_array = []
for i in images:
	b = np.reshape(i, (1,np.product(i.shape)))
	final_array.append(b)

images = final_array
hamming_values = []
values = []
for i in range(len(names)):
	names[i] = names[i][42:],

for i in range (len(images)):
	value1 = images[i]
	for j in range (0, len(images)):
		if j == i:
			continue
		value2 = images[j]

		values.append("%.2f" % hamming_distance(value1, value2))
	hamming_values.append(values)
	values = []
for i in range(len(hamming_values)):
	print(names[i][0], hamming_values[i])

# print(hamming_distance(final_array[0], final_array[1]))