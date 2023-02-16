import jieba
import jieba.posseg as pseg
from opencc import OpenCC
import csv


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


def find_dialogue(sentence):
    text = '「对话内容」'
    return sentence.find(text)


def word_embedding(document, Word_List):
    ListToLSTM = []
    Speaker = []
    i = 1
    for sentence in document:
        loc_List = []
        print('{}. ------------------'.format(i))
        sentence = tradition_to_simple(sentence)
        dialogue_position = find_dialogue(sentence)
        sentence = sentence.replace('「对话内容」', '')
        load_dict()
        seg_list = pseg.cut(sentence)
        flag_list = []
        word_list = []
        for word, flag in seg_list:
            if word is not " ":
                if word == '道':
                    flag = 'v'
                if word == '因':
                    flag = 'c'
                if word == '忙':
                    flag = 'd'
                if word == '问':
                    flag = 'v'
                if word == '命':
                    flag = 'v'
                flag_list.append(flag)
                word_list.append(word)
                print(f'{word} {flag}')
        print(flag_list)
        if embedding_correct():
            for word in word_list:
                if word not in Word_List:
                    Word_List.append(word)
                loc_List.append(Word_List.index(word))
            ListToLSTM = join_the_LSTMList(ListToLSTM, sentence, flag_list, dialogue_position, loc_List)
            Speaker = join_the_SpeakerList(Speaker, input('請輸入說話者'))
            # input csv_list
        else:
            correct_embedding = get_the_correct_embedding(sentence)
            update_dictionary()
            fixed_seg_list = create_seg_list()
            for word in fixed_seg_list:
                if word not in Word_List:
                    Word_List.append(word)
                loc_List.append(Word_List.index(word))
            ListToLSTM = join_the_LSTMList(ListToLSTM, sentence, correct_embedding, dialogue_position, loc_List)
            Speaker = join_the_SpeakerList(Speaker, input('請輸入說話者'))
        user_input = input('是否繼續?(Y/N)')
        if user_input.upper() == 'Y':
            pass
        elif user_input.upper() == 'N':
            print('目前進行到第 {} 句，期待下次繼續進行。'.format(i))
            break
        else:
            print('輸入錯誤，將繼續進行')
        i += 1
    return ListToLSTM, Speaker, Word_List


def update_dictionary():
    user_input = input('請輸入需要添加的字典內容，若不添加請輸入0')
    while user_input != '0':
        with open('./dictionary.txt', "a", encoding='UTF-8') as file:
            file.write(user_input+'\n')
        user_input = input('請輸入需要添加的字典內容，若不添加請輸入0')


def embedding_correct():
    user_input = input('請問斷詞是否正確?(Y/N)')
    if user_input.upper() == 'Y':
        return True
    else:
        return False


def get_the_correct_embedding(oringin_text):
    print(oringin_text)
    correct_embedding = []
    while True:
        user_input = input('請輸入正確詞性斷詞結果:(part of speech part of speech)')
        print(user_input)
        reconfirm = input('請問是否正確?(Y/N)')
        if reconfirm.upper() == 'Y':
            if " " in user_input:
                correct_embedding = user_input.split()
                return correct_embedding
            else:
                pass


def join_the_LSTMList(ListToLSTM, text, embedding_list, dialogue_position, Loc_List):
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

    if dialogue_position == 0:
        flag_list.insert(0, 99)
        flag_list.append(0)
    else:
        flag_list.append(99)

    for loc in Loc_List:
        flag_list.append(loc)

    if dialogue_position == 0:
        pass
    else:
        flag_list.append(0)
    flag_list.insert(0, text)
    ListToLSTM.append(flag_list)
    return ListToLSTM


def join_the_SpeakerList(Speaker, position):
    Speaker.append(position)
    return Speaker


def WriteToCSVFile(path, ListToLSTM):
    with open(path, 'a', newline='', encoding='utf-8-sig') as OutputFile:
        OutputWriter = csv.writer(OutputFile, delimiter=',')
        for Record in ListToLSTM:
            OutputWriter.writerow(Record)


def fix_the_LSTM_list_length(List):
    max_length = 40
    for sub_list in List:
        if len(sub_list) < max_length:
            for i in range(0, (max_length-len(sub_list))):
                sub_list.append(0)


def update_LOC(word_loc):
    with open(r'./LOC.txt', "w", encoding='UTF-8') as file:
        for string in word_loc:
            file.write(string + '\n')


def create_seg_list():
    seg_list = []
    user_input = input('請輸入正確斷詞結果，空格隔開')
    if " " in user_input:
        seg_list = user_input.split()
    return seg_list



documents = load_documents(r'./RM2_toEmbed.txt')
documents = replaceLine(documents)
word_Loc = load_documents(r'./LOC.txt')
word_Loc = replaceLine(word_Loc)
print(len(word_Loc))
print(word_Loc[-1:])
# LSTM_List, Speaker_List, word_Loc = word_embedding(documents, word_Loc)
#
# print('---LSTM_List---')
# print(LSTM_List[:5])
# print('---Speaker_List---')
# print(Speaker_List[:5])
# print('---Word_Loc---')
# print(word_Loc)
# update_LOC(word_Loc)
# WriteToCSVFile(r'./rm_loc_csv.csv', LSTM_List)
# WriteToCSVFile(r'./speaker2.csv', Speaker_List)


