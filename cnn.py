import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from sklearn.model_selection import train_test_split, StratifiedKFold

DATADIR = os.getcwd()
path = os.path.join(DATADIR, 'Chinese')
IMG_SIZE = 80
NUM_CLASSES = 2

for folder in os.listdir(path):

	if folder == '.DS_Store':
		continue

	# get X and y from the pickled dataset
	pickle_in = open(os.path.join(path, folder, 'X.pickle'), 'rb')
	X = pickle.load(pickle_in)
	pickle_in.close()

	pickle_in = open(os.path.join(path, folder, 'y.pickle'), 'rb')
	y = pickle.load(pickle_in)
	pickle_in.close()

	# divide into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0)

	# reshape for entering into the network
	X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
	X_test = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
	input_shape = (IMG_SIZE, IMG_SIZE, 1)

	# change to float type to divide by 255 later
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# normalize
	X_train /= 255
	X_test /= 255

	y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
	y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

	# make the model and train it
	model = tf.keras.models.Sequential()

	model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Conv2D(16, (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Dropout(0.25))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
	model.add(Dropout(0.5))
	model.add(tf.keras.layers.Dense(NUM_CLASSES, activation = tf.nn.softmax))

	model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model.fit(X_train, y_train, epochs = 100)

	# val_loss, val_acc = model.evaluate(X_test, y_test)
	# print("Folder:", folder, "Loss:", val_loss, "Accuracy:", val_acc)

	# save model in JSON
	json_model = model.to_json()
	open('model_architecture.json', 'w').write(json_model)
	
	# save weights INSIDE EACH FOLDER
	model.save_weights(os.path.join(path, folder, 'model_weights.h5'), overwrite=True)

	break




















