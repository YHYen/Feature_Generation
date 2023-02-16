import keras.utils
import numpy as np
import tensorflow
from sklearn.metrics import accuracy_score, f1_score
from pandas import read_csv
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import MaxPooling1D, Conv1D, Dense, Embedding, Flatten, Dropout


dataset = read_csv(r'./dataset4.csv', delimiter=",", header=None)

print(dataset.head(5))

values = dataset.values

values_X, values_Y = values[:, :-1], values[:, -1]
print(values_X.shape)
print(values_Y.shape)

# reshape train, test data
train_size = int(len(values_Y) * 0.9)
test_size = len(values_Y) - train_size
train_X, test_X = values_X[0:train_size, :], values_X[train_size:len(values_Y), :]
train_Y, test_Y = values_Y[0:train_size], values_Y[train_size:len(values_Y)]
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
max_len = len(train_X[0])

# design network
inputs = Input(name = 'inputs', shape=[max_len])
# Embedding(length of word list, batch_size, max_len)
layer = Embedding(1905, 128, input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(35, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])
# optimizer can be 'adam'


# main_input = Input(shape=(50,), dtype='float64')
#
# embedder = Embedding(1905, 16, input_length=50, trainable=False)
# embed = embedder(main_input)
#
# cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
# cnn1 = MaxPooling1D(pool_size=48)(cnn1)
# cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
# cnn2 = MaxPooling1D(pool_size=47)(cnn2)
# cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
# cnn3 = MaxPooling1D(pool_size=46)(cnn3)
# # concat_sc = Concatenate(axis=-1)([sc1, sc2])
# cnn = tensorflow.keras.layers.Concatenate(axis=-1)([cnn1, cnn2, cnn3])
# flat = Flatten()(cnn)
# drop = Dropout(0.1)(flat)
# main_output = Dense(3, activation='softmax')(drop)
# model = Model(inputs=main_input, outputs=main_output)
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

one_hot_labels_Train = keras.utils.to_categorical(train_Y, num_classes=35)
one_hot_labels_Test = keras.utils.to_categorical(test_Y, num_classes=35)

model.fit(train_X, one_hot_labels_Train, batch_size=16, epochs=100, validation_data=(test_X, one_hot_labels_Test))
# , callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]

# result = model.predict(test_X)
# result_labels = np.argmax(result, axis=1)
# predict_Y = list(map(str, result_labels))
# print('accuracy', accuracy_score(test_Y, predict_Y))
# print('f1_Score', f1_score(test_Y, predict_Y, average='weighted'))
