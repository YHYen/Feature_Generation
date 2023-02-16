import csv
from pandas import read_csv
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


def ChangeDataTypeToInteger(List):
    for i in range(len(List)):
        List[i] = int(List[i])
    return List


def RetypeLabelList(List):
    for i in range(len(List)):
        List[i] = List[i][0]
    return List


validationSet = ReadFromCSVFile(r'./Dataset3L/FinalPOSTestset2.csv')
validationSet = SampleListRetype(validationSet)
validationSet = LeaveFeatureList(validationSet)
validation_X = RetypeToNumpyNDArray(validationSet)
validation_X = validation_X.reshape((validation_X.shape[0], validation_X.shape[1], -1))
print(validation_X.shape)

validationSetLabel = ReadFromCSVFile(r'./Dataset3L/newLabel2.csv')
validationSetLabel.pop(0)
validationSetLabel = RetypeLabelList(validationSetLabel)
validationSetLabel = ChangeDataTypeToInteger(validationSetLabel)
print(validationSetLabel[0:5])

model = tf.contrib.keras.models.load_model(r'./model_save3L/newModel_Pos_150E.h5')

predict = model.predict(validation_X)
predict_Label = []

for pre in predict:
    maximum = 0
    for index in range(len(pre)):
        if pre[maximum] < pre[index]:
            maximum = index
    predict_Label.append(maximum)
print(len(predict_Label))

predict_Label = ChangeDataTypeToInteger(predict_Label)

correctAnswer = 0
wrongAnswer = 0
labelZeroWrong = 0
labelOneWrong = 0
labelTwoWrong = 0
labelThreeWrong = 0
labelFourWrong = 0

for index in range(len(predict_Label)):
    if predict_Label[index] == validationSetLabel[index]:
        correctAnswer += 1
    else:
        wrongAnswer += 1
        if validationSetLabel[index] == 0:
            labelZeroWrong += 1
            print(index)
            print(predict[index])
        elif validationSetLabel[index] == 1:
            labelOneWrong += 1
        elif validationSetLabel[index] == 2:
            labelTwoWrong += 1
        elif validationSetLabel[index] == 3:
            labelThreeWrong += 1
        elif validationSetLabel[index] == 4:
            print(index)
            print(predict[index])
            labelFourWrong += 1

print('Correct Answer ', correctAnswer)
print('Wrong Answer ', wrongAnswer)
print('Label Zero Wrong: ', labelZeroWrong)
print('Label One Wrong: ', labelOneWrong)
print('Label Two Wrong: ', labelTwoWrong)
print('Label Three Wrong: ', labelThreeWrong)
print('Label Four Wrong:', labelFourWrong)