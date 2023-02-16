import csv
import jieba
import jieba.posseg as pseg
from opencc import OpenCC


def load_dict():
    jieba.load_userdict('./dictionary.txt')


def load_documents(path):
    with open(path, encoding='utf-8') as TextFile:
        text_list = TextFile.readlines()
        replaceLine(text_list)
    return text_list


def replaceLine(TextList):
    for i in range(0, len(TextList)):
        TextList[i] = TextList[i].replace('\n', '')
    return TextList


def tradition_to_simple(text):
    cc = OpenCC('t2s')
    text = cc.convert(text)
    return text


def join_the_LSTMList(ListToLSTM, embedding_list):
    pos = ['ccccc', 'a', 'ad', 'ag', 'an', 'b', 'c', 'd', 'df', 'dg', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'mg', 'mq', 'n', 'ng', 'nr', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'rg', 'rr', 'rz', 's', 't', 'tg', 'u', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz', 'v', 'vd', 'vg', 'vi', 'vn', 'vq', 'x', 'y', 'z', 'zg']
    flag_list = []
    for postag in embedding_list:
        have_pos = False
        for p in pos:
            if postag == p:
                have_pos = True
                flag_list.append(pos.index(postag))
        if not have_pos:
            flag_list.append(97)
    flag_list.append(99)
    ListToLSTM.append(flag_list)
    return ListToLSTM


def WriteToCSVFile(path, ListToLSTM):
    with open(path, 'w', newline='', encoding='utf-8-sig') as OutputFile:
        OutputWriter = csv.writer(OutputFile, delimiter=',')
        for Record in ListToLSTM:
            OutputWriter.writerow(Record)


def wordEmbedding(textList, wordList):
    ListToLSTM = []
    locationList = []
    for sentence in textList:
        sentence = tradition_to_simple(sentence)
        sentence = sentence.replace('「对话内容」', '')
        load_dict()
        seg_list = pseg.cut(sentence)
        flag_list = []
        word_list = []
        loc_list = []
        for word, flag in seg_list:
            if word is not " ":
                flag_list.append(flag)
                word_list.append(word)
        for word in word_list:
            if word not in wordList:
                loc_list.append(0)
            else:
                loc_list.append(wordList.index(word))
        loc_list.append(1904)
        locationList.append(loc_list)
        ListToLSTM = join_the_LSTMList(ListToLSTM, flag_list)
    return ListToLSTM, locationList


def fix_the_LSTM_list_length(List):
    max_length = 33
    for sub_list in List:
        if len(sub_list) < max_length:
            for i in range(0, (max_length-len(sub_list))):
                sub_list.append(0)


documents = load_documents(r'./dialogueData.txt')
documents = replaceLine(documents)
word_Loc = load_documents(r'./LOC.txt')
word_Loc = replaceLine(word_Loc)
LSTM_List, LOC_List = wordEmbedding(documents, word_Loc)
fix_the_LSTM_list_length(LSTM_List)
WriteToCSVFile(r'./final/dialogueData.csv', LSTM_List)
WriteToCSVFile(r'./final/locationData.csv', LOC_List)
