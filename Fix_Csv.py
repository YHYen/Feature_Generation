from main import ReadFromCSVFile, WriteToCSVFile, fix_the_LSTM_list_length, all_to_number


def change_label_to_Fourth(List):
    for speaker in List:
        speaker[0] = int(int(speaker[0]) / 11)
    return List


def change_Label_By_Split_Comma(List, Speaker):
    Label = []
    Location = []
    POS = []
    for i in range(len(List)):
        Pos_List, Location_List = find_Location_List(List[i])
        POS.append(Pos_List)
        Location.append(Location_List)
        if Speaker[i][0] == '0':
            Label.append(0)
        elif Speaker[i][0] == '34':
            Label.append(4)
        else:
            Comma_Index_List = find_Comma_Index(Location_List)
            for j in range(len(Comma_Index_List)):
                label = 0
                if int(Speaker[i][0]) > Comma_Index_List[j]:
                    label = j
            Label.append(label)
    return POS, Location, Label


def find_Location_List(List):
    Location_List_start = List.index('99')
    Location_List = List[Location_List_start+1:]
    POS_List = List[:Location_List_start+1]
    return POS_List, Location_List


def find_Comma_Index(List):
    Comma_Index_List = [0]
    index = 0
    while True:
        try:
            Comma_Index = List.index('19', index)
            Comma_Index_List.append(Comma_Index)
            index = Comma_Index+1
        except ValueError:
            break
    return Comma_Index_List


def delete_the_sentence(List):
    for i in range(0, len(List)):
        del List[i][0]
    return List


def append_the_answer(List, label):
    for i in range(0, len(List)):
        List[i].append(label[i])
    return List


loc_list = ReadFromCSVFile(r'./rm_loc_csv.csv')
speaker_list2 = ReadFromCSVFile(r'./speaker2.csv')
# speaker_list2 = change_label_to_Fourth(speaker_list2)
POS, LOCATION, LABEL = change_Label_By_Split_Comma(loc_list, speaker_list2)
POS = delete_the_sentence(POS)
POS = fix_the_LSTM_list_length(POS)
LOCATION = fix_the_LSTM_list_length(LOCATION)
POS = append_the_answer(POS, LABEL)
LOCATION = append_the_answer(LOCATION, LABEL)
POS = all_to_number(POS)
LOCATION = all_to_number(LOCATION)
print(POS[:5])
print(LOCATION[:5])
print(len(POS[0]))
# WriteToCSVFile(r'./dataset_POS.csv', POS)
# WriteToCSVFile(r'./dataset_LOC.csv', LOCATION)



# 針對 標點符號做事 位初期標記、都是直接寫第一個人，所以沒有、變成標點符號的問題，就算有，判斷的也跟他無關。
# 這樣就可以很好的區分Label
# 換個想法，如果直接使用後半部Location來進行區分，其實就不用擔心判斷錯標點符號的問題了
