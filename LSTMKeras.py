from pandas import read_csv
import os
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


dataset = read_csv(r'./Dataset/200DatasetForFourLable_LOC.csv', delimiter=",", header=None)

print(dataset.head(5))

values = dataset.values

values_X, values_Y = values[:, :-1], values[:, -1]
print(values_X.shape)
print(values_Y.shape)
print(type(values_X[0]))

# reshape train, test data
train_size = int(len(values_Y) * 0.9)
test_size = len(values_Y) - train_size
test_X = values_X[train_size:len(values_Y), :]
test_Y = values_Y[train_size:len(values_Y)]
train_X = values_X.reshape((values_X.shape[0], values_X.shape[1], -1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], -1))
print(train_X.shape, values_Y.shape, test_X.shape, test_Y.shape)
print(train_X.shape[2])
print(type(train_X))
print(type(test_X))

# model = models.Sequential()
# model.add(layers.Conv1D(50, 2, padding="same", activation='relu', input_shape=(train_X.shape[1], 1)))
# model.add(layers.MaxPooling1D(2, padding="same"))
# model.add(layers.Conv1D(50, 2, padding="same", activation='relu'))
# model.add(layers.MaxPooling1D(2, padding="same"))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(5))
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(train_X, values_Y, batch_size=8, epochs=300, validation_data=(test_X, test_Y))
# #
# plt.plot(history.history['acc'], label='accuracy')
# plt.plot(history.history['val_acc'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.99, 1])
# plt.legend(loc='lower right')
# plt.show()
#
# test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
# #
# model.save(r"./model_save/model_LOC_4L_300E.h5")
