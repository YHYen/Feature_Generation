import csv
from pandas import read_csv
import os
import numpy
import math
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


def ReadFromCSVFile(path):
    Data_List = []
    with open(path, newline='', encoding='utf-8-sig') as CSVFile:
        # csv.reader(csvFile, delimiter='[,: ]')
        DataRows = csv.reader(CSVFile)
        for row in DataRows:
            Data_List.append(row)
    return Data_List


def WriteToCSVFile(path, List):
    with open(path, 'a', newline='', encoding='utf-8-sig') as OutputFile:
        OutputWriter = csv.writer(OutputFile, delimiter=',')
        for Record in List:
            OutputWriter.writerow(Record)


def SampleListRetype(sampleList):
    for sample in sampleList:
        sample[0] = int(sample[0])
        sample[1] = ListStringToList(sample[1])
        sample[1] = SampleResize(sample[1])
        for i in range(len(sample[1])):
            sample[1][i] = float(sample[1][i])
    return sampleList


def SampleResize(sample):
    while len(sample) < 33:
        sample.append(0.0)
    return sample


def ListStringToList(listString):
    res = listString.strip('][').split(', ')
    return res


def RetypeToNumpyNDArray(featureList):
    for i in range(len(featureList)):
        featureList[i] = numpy.array(featureList[i])
    featureList = numpy.array(featureList)
    return featureList


def LeaveFeatureList(sampleList):
    featureList = []
    for i in range(len(sampleList)):
        featureList.append(sampleList[i][1])
    return featureList


trainset = ReadFromCSVFile(r'./Dataset3L/FinalTrainSetPOS6.csv')
trainset = SampleListRetype(trainset)
trainset = LeaveFeatureList(trainset)
train_X = RetypeToNumpyNDArray(trainset)
print(train_X.shape)

trainsetLabel = read_csv(r'./Dataset3L/FinalTrainSetLabel.csv', delimiter=",", header=None)
train_Y = trainsetLabel.values
train_Y = train_Y.reshape((train_Y.shape[0]))
print(train_Y.shape)

validationSet = ReadFromCSVFile(r'./Dataset3L/FinalPOSTestset.csv')
validationSet = SampleListRetype(validationSet)
validationSet = LeaveFeatureList(validationSet)
validation_X = RetypeToNumpyNDArray(validationSet)
print(validation_X.shape)

validationSetLabel = read_csv(r'./Dataset3L/newLabel.csv', delimiter=",")
validation_Y = validationSetLabel.values
validation_Y = validation_Y.reshape(validation_Y.shape[0])
print(validation_Y.shape)

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], -1))
validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[1], -1))
print(train_X.shape, train_Y.shape, validation_X.shape, validation_Y.shape)

# Create model
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
model.add(layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_X, train_Y, batch_size=8, epochs=150, validation_data=(validation_X, validation_Y))
#
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(validation_X, validation_Y, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
model.save(r"./model_save3L/newModel_Pos_150E_6.h5")
