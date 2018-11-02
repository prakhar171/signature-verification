import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2, os, pickle, random

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, 'Chinese', category)
		class_num = CATEGORIES.index(category)
		
		for folder in os.listdir(path):

			if folder == '.DS_Store':
				continue

			images_path = os.path.join(path, folder)

			try:
				for img in os.listdir(images_path):
					img_array = cv2.imread(os.path.join(images_path, img), cv2.IMREAD_GRAYSCALE)
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

					training_data.append([new_array, class_num])
			
			except Exception as e:
				# print("Prakhar tatti")
				print(e)
				# pass


DATADIR = os.getcwd()
CATEGORIES = ['Forgeries', 'Genuine']
IMG_SIZE = 50

training_data = []
create_training_data()

# shuffle the data to avoid bias
random.shuffle(training_data)

# make input and output arrays
X = []
y = []

for img, label in training_data:
	X.append(img)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# store as pickle file for later
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()