# Load modules
import csv
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utilities import process_img
from utilities import mirror_data
from utilities import augment_brightness
from utilities import generate_batch

# Build image and steering angle data
X_train = []
y_train = []

with open("driving_log.csv", "rt") as f:
	reader = csv.reader(f, skipinitialspace=True)
	data = np.array([row for row in reader])

for i in range(len(data)):
	X_train.append(data[i][0])
	X_train.append(data[i][1])
	X_train.append(data[i][2])
	y_train.append(float(data[i][3]))
	y_train.append(float(data[i][3]) + 0.25)
	y_train.append(float(data[i][3]) - 0.25)

data_expanded = np.column_stack((X_train, y_train))

# Shuffle data and split into validation set
data_shuffle = shuffle(data_expanded)
data_train, data_validation = train_test_split(data_shuffle, test_size=0.2, random_state=879345)

# Build model
model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 1), border_mode='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3, border_mode='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164, W_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(100, W_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50, W_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='mse')

# Load the weights if they exist
try:
	model.load_weights('model.h5')
except:
	pass

# Train the model
history = model.fit_generator(generate_batch(data_train, batch_size=128), 
											 len(data_train), nb_epoch=1, 
											 validation_data=generate_batch(data_validation, batch_size=128),
											 nb_val_samples=len(data_validation))

# Save the model and weights
model.save_weights('model.h5')
json = model.to_json()
with open('model.json', 'w') as f:
	f.write(json)