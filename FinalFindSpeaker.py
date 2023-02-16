import csv
from opencc import OpenCC


def load_documents(path):
    with open(path, encoding='utf-8') as TextFile:
        text_list = TextFile.readlines()
        replaceLine(text_list)
    return text_list


def replaceLine(TextList):
    for i in range(0, len(TextList)):
        TextList[i] = TextList[i].replace('\n', '')
    return TextList


def simple_to_traditional(list):
    cc = OpenCC('s2t')
    for i in range(len(list)):
        list[i] = cc.convert(list[i])
    return list


def ReadFromCSVFile(path):
    Data_List = []
    with open(path, newline='', encoding='utf-8-sig') as CSVFile:
        # csv.reader(csvFile, delimiter='[,: ]')
        DataRows = csv.reader(CSVFile)
        for row in DataRows:
            Data_List.append(row)
    return Data_List


def WriteTextFile(path, List):
    with open(path, 'w', newline='', encoding='utf-8-sig') as outputFile:
        for name in List:
            outputFile.write(name + '\n')


def RetypeLabelList(List):
    for i in range(len(List)):
        List[i] = List[i][0]
    return List


def findSpeaker(posList, locList, labelList, wordLoc):
    speaker = []
    speakerIndex = findSpeakerIndex(labelList, posList, locList)
    print(speakerIndex)
    for i in range(len(speakerIndex)):
        if speakerIndex[i] == 9999:
            speaker.append('not Found1')
        elif speakerIndex[i] == 9998:
            speaker.append('not found2')
        elif speakerIndex[i] == 34:
            speaker.append('last')
        else:
            speaker.append(wordLoc[int(locList[i][speakerIndex[i]])])
    return speaker


def find_Comma_Index(List):
    index = 0
    while True:
        try:
            Comma_Index = List.index('19', index)
            index = Comma_Index+1
        except ValueError:
            break
    return index


def findSpeakerIndex(labelList, posList, locList):
    speakerIndex = []
    nr = 9998
    for i in range(len(labelList)):
        if labelList[i] == '0':
            commaIndex = find_Comma_Index(locList[i])
            nr = speakerIsNROrN(commaIndex, posList[i])
        elif labelList[i] == '1':
            index = 0
            nr = speakerIsNROrN(index, posList[i])
            while True:
                try:
                    Comma_Index = posList[i].index('50', index)
                except ValueError:
                    break
                if posList[i][Comma_Index+1:Comma_Index+3] == ['30', '21'] or posList[i][Comma_Index+1:Comma_Index+3] == ['30', '23']:
                    nr = Comma_Index + 2
                elif posList[i][Comma_Index+1] == '21' or posList[i][Comma_Index+1] == '23':
                    nr = Comma_Index + 1
                index = Comma_Index + 1

        elif labelList[i] == '2':
            nr = 34
        speakerIndex.append(nr)
    return speakerIndex


def speakerIsNROrN(commaIndex, posList):
    try:
        nr = posList.index('23', commaIndex)
    except ValueError:
        try:
            nr = posList.index('21', commaIndex)
        except ValueError:
            nr = 9999
    return nr


POS_List = ReadFromCSVFile(r'./final/dialogueData.csv')
LOC_List = ReadFromCSVFile(r'./final/locationData.csv')
Label_List = ReadFromCSVFile(r'./final/finalLabel.csv')
Label_List = RetypeLabelList(Label_List)
word_Loc = load_documents(r'./LOC.txt')
word_Loc = replaceLine(word_Loc)
speakerList = findSpeaker(POS_List, LOC_List, Label_List, word_Loc)
speakerList = simple_to_traditional(speakerList)
# print(speakerList)
WriteTextFile(r'./final/ResultName.txt', speakerList)


