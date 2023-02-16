import csv
import numpy
import tensorflow as tf


def ReadFromCSVFile(path):
    Data_List = []
    with open(path, newline='', encoding='utf-8-sig') as CSVFile:
        # csv.reader(csvFile, delimiter='[,: ]')
        DataRows = csv.reader(CSVFile)
        for row in DataRows:
            Data_List.append(row)
    return Data_List


def SampleListRetype(sampleList):
    for sample in sampleList:
        for i in range(len(sample)):
            sample[i] = float(sample[i])
    return sampleList


def RetypeToNumpyNDArray(featureList):
    for i in range(len(featureList)):
        featureList[i] = numpy.array(featureList[i])
    featureList = numpy.array(featureList)
    return featureList


def WriteToCSVFile(path, ListToLSTM):
    with open(path, 'w', newline='', encoding='utf-8-sig') as OutputFile:
        OutputWriter = csv.writer(OutputFile, delimiter=',')
        for Record in ListToLSTM:
            OutputWriter.writerow(str(Record))


validationSet = ReadFromCSVFile(r'./final/dialogueData.csv')
print(validationSet)
print(type(validationSet))
validationSet = SampleListRetype(validationSet)
validation_X = RetypeToNumpyNDArray(validationSet)
validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[1], -1))
print(validationSet[0])
print(type(validationSet[0]))
print(validation_X.shape)

model = tf.contrib.keras.models.load_model(r'./model_save3L/newModel_POS_150E.h5')

predict = model.predict(validation_X)
predict_Label = []

for pre in predict:
    maximum = 0
    for index in range(len(pre)):
        if pre[maximum] < pre[index]:
            maximum = index
    predict_Label.append(maximum)

WriteToCSVFile(r'./final/finalLabel.csv', predict_Label)
