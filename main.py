from opencc import OpenCC
import jieba
import jieba.posseg as pseg
import csv

# text = '石頭笑答道：「對話內容」'
# find_text = '對話內容'
# cc = OpenCC('t2s')
# text = cc.convert(text)
# print(text.find(find_text))

# pos = ['x', 'n', 'nr', 'nz', 'a', 'm', 'c', 'PER', 'f', 'ns', 'v', 'ad', 'q', 'u', 'LOC', 's', 'nt', 'vd', 'an', 'r', 'xc', 'ORG', 't', 'nw', 'vn', 'd', 'p', 'w', 'TIME']
# postag = 'nr'
# print(pos.index(postag))


# def fix_the_list_length(List):
#     max_length = find_max_list_length(List)
#     for sub_list in List:
#         if len(sub_list) < max_length:
#             for i in range(0, (max_length-len(sub_list))):
#                 sub_list.append(0)
#
#
# def find_max_list_length(List):
#     maximum = len(List[0])
#     for sub_list in List:
#         if len(sub_list) > maximum:
#             maximum = len(sub_list)
#     return maximum
#
# text = 't nr v ul nr x d v v v x'
# list1 = ['t', 'nr', 'v', 'ul', 'nr', 'x', 'd', 'v', 'v', 'v', 'x']
#

# text = 'a ad ag an b c d df dg e f g h i j k l m mg mq n ng nr ns nt nz o p q r rg rr rz s t tg u ud ug uj ul uv uz v vd vg vi vn vq x y z zg'
# list2 = text.split()
# list = []
# position = 11
# list.append(int(input('請輸入說話者')))
# list.append(12)
# print(list)

def ReadFromCSVFile(path):
    Data_List = []
    with open(path, newline='', encoding='utf-8-sig') as CSVFile:
        # csv.reader(csvFile, delimiter='[,: ]')
        DataRows = csv.reader(CSVFile)
        for row in DataRows:
            Data_List.append(row)
    return Data_List


def fix_the_LSTM_list_length(List):
    max_length = find_the_max_length(List)
    for sub_list in List:
        sub_list[-1:] = (1904, )
        if len(sub_list) < max_length:
            for i in range(0, (max_length-len(sub_list))):
                sub_list.append(0)
    return List


def find_the_max_length(List):
    max = len(List[0])
    for sub_list in List:
        if max < len(sub_list):
            max = len(sub_list)
    print(max)
    return max


def append_the_answer(List, speaker):
    for i in range(0, len(List)):
        del List[i][0]
        List[i].append(int(speaker[i][0]))
    return List


def all_to_number(List):
    for sub_list in List:
        for i in range(0, len(sub_list)):
            sub_list[i] = int(sub_list[i])
    return List


def WriteToCSVFile(path, List):
    with open(path, 'a', newline='', encoding='utf-8-sig') as OutputFile:
        OutputWriter = csv.writer(OutputFile, delimiter=',')
        for Record in List:
            OutputWriter.writerow(Record)

#
# csv_list = ReadFromCSVFile(r'./rm_loc_csv.csv')
# speaker_list = ReadFromCSVFile(r'./speaker2.csv')
# csv_list = fix_the_LSTM_list_length(csv_list)
# csv_list = append_the_answer(csv_list, speaker_list)
# csv_list = all_to_number(csv_list)
# print(csv_list[:5])
# WriteToCSVFile(r'./dataset4.csv', csv_list)


# 1904 對話內容 
# 答案要加1

