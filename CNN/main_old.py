import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

# get X and y from the created dataset
pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

# normalize the data
X = tf.keras.utils.normalize(X, axis = 1)


# divide into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation = tf.nn.sigmoid))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10)

# save model in JSON
json_model = model.to_json()
open('model_architecture.json', 'w').write(json_model)
# saving weights
model.save_weights('model_weights.h5', overwrite=True)

new_model = tf.keras.models.model_from_json(open('model_architecture.json').read())
new_model.load_weights('model_weights.h5')

predictions = new_model.predict(X_test)

print(predictions)
