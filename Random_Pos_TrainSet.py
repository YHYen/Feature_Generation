import random

from main import ReadFromCSVFile, WriteToCSVFile, fix_the_LSTM_list_length, all_to_number


def deleteLabelData(dataset):
    datasetWithoutLabel = []
    for i in range(0, len(dataset)):
        data = dataset[i][:-1]
        data = replaceDataToNumber(data)
        datasetWithoutLabel.append(data)
    return datasetWithoutLabel


def replaceDataToNumber(data):
    dataAfterTransfer = []
    for i in range(0, len(data)):
        dataAfterTransfer.append(float(data[i]))
    return dataAfterTransfer


def addSerialNumber(dataset):
    datasetWithSerialNumber = []
    for i in range(0, len(dataset)):
        data = [str(i+1), str(dataset[i])]
        datasetWithSerialNumber.append(data)
    return datasetWithSerialNumber


def createSerialSetForAllLabel(dataset):
    labelZeroSet, labelOneSet, labelTwoSet = [], [], []
    for i in range(0, len(dataset)):
        if dataset[i][0] == '0':
            labelZeroSet.append(i)
        if dataset[i][0] == '1':
            labelOneSet.append(i)
        if dataset[i][0] == '2':
            labelTwoSet.append(i)
    return labelZeroSet, labelOneSet, labelTwoSet

def chooseRandomData(labelZeroSet, labelOneSet, labelTwoSet):
    randomLabelZeroSet = random.sample(labelZeroSet, 125)
    randomLabelOneSet = random.sample(labelOneSet, 125)



purpose_Dataset = ReadFromCSVFile(r'./Dataset3L/FinalTrainSetPOS.csv')
print('--------Purpose---------')
print(purpose_Dataset[:5])
print('=========================')
pos_Dataset = ReadFromCSVFile(r'./dataset_POS.csv')
pos_Dataset = deleteLabelData(pos_Dataset)
pos_DatasetWithSerialNumber = addSerialNumber(pos_Dataset)
print(pos_DatasetWithSerialNumber[:5])

labelSet = ReadFromCSVFile(r'./Dataset3L/newLabel.csv')
labelZero, labelOne, labelTwo = createSerialSetForAllLabel(labelSet)
# print(labelZero[:5])
# print(len(labelZero))
# print(len(labelOne))
# print(len(labelTwo))
