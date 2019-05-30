import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from sklearn.model_selection import train_test_split

DATADIR = os.getcwd()
path = os.path.join(DATADIR, 'Chinese')

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

	# normalize the data
	X = tf.keras.utils.normalize(X, axis = 1)

	# divide into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

	# make the model and train it
	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))

	model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model.fit(X_train, y_train, epochs = 75)
	val_loss, val_acc = model.evaluate(X_test, y_test)

	print("Folder:", folder, "Loss:", val_loss, "Accuracy:", val_acc)




















