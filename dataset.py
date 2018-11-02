import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2, os, pickle, random

'''
# class_num
	1 -> genuine
	0 -> forged
'''

def create_training_data(folder):
	images_path = os.path.join(path, folder)

	training_data = []

	for img in os.listdir(images_path):

		try:
			img_array = cv2.imread(os.path.join(images_path, img), cv2.IMREAD_GRAYSCALE)
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

			class_number = 1
			if (len(img) > 10):
				class_number = 0

			for i in range(1):
				training_data.append([new_array, class_number])

		except Exception as e:
			# print("Prakhar tatti")
			pass

	# shuffle the data to avoid bias
	random.shuffle(training_data)

	# make input and output arrays
	X = []
	y = []

	for img, label in training_data:
		X.append(img)
		y.append(label)

	# change to numpy array and change shape to facilitate flattening
	# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	X = np.array(X)
	y = np.array(y)

	# store dataset as pickle file for later use
	pickle_out = open(os.path.join(images_path, 'X.pickle'), 'wb')
	pickle.dump(X, pickle_out)
	pickle_out.close()

	pickle_out = open(os.path.join(images_path, 'y.pickle'), 'wb')
	pickle.dump(y, pickle_out)
	pickle_out.close()

	print(folder, "dataset created", len(training_data))


DATADIR = os.getcwd()
path = os.path.join(DATADIR, 'Chinese')
IMG_SIZE = 80

for folder in os.listdir(path):

	if folder == '.DS_Store':
		continue

	create_training_data(folder)