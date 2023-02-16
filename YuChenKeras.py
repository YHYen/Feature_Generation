from pandas import read_csv
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


dataset_X = read_csv(r'./YuChen/train_data_x_v3.csv', delimiter=",", header=None)
print(dataset_X.head(5))
values_X = dataset_X.values
dataset_Y = read_csv(r'./YuChen/train_data_y_v3.csv', delimiter=",", header=None)
print(dataset_Y.head(5))
values_Y = dataset_Y.values
train_Y = values_Y.reshape((values_Y.shape[0]))
print(values_X.shape)
print(train_Y.shape)

testSet_X = read_csv(r'./YuChen/test_data_x_v2.csv', delimiter=",", header=None)
print(testSet_X.head(5))
validationSet_X = testSet_X.values
testSet_Y = read_csv(r'./YuChen/test_data_y_v2.csv', delimiter=",", header=None)
print(testSet_Y.head(5))
validationSet_Y = testSet_Y.values
validation_Y = validationSet_Y.reshape(validationSet_Y.shape[0])
print(validationSet_X.shape)
print(validation_Y.shape)

train_X = values_X.reshape((values_X.shape[0], values_X.shape[1], -1))
validation_X = validationSet_X.reshape((validationSet_X.shape[0], validationSet_X.shape[1], -1))
print(train_X.shape, train_Y.shape, validation_X.shape, validation_Y.shape)

model = models.Sequential()
model.add(layers.Conv1D(50, 2, padding="same", activation='relu', input_shape=(train_X.shape[1], 1)))
model.add(layers.MaxPooling1D(2, padding="same"))
model.add(layers.Conv1D(50, 2, padding="same", activation='relu'))
model.add(layers.MaxPooling1D(2, padding="same"))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_X, train_Y, batch_size=8, epochs=300, validation_data=(validation_X, validation_Y))

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.99, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(validation_X, validation_Y, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

model.save(r"./YuChen/300Ev3.h5")
