from pandas import read_csv
import tensorflow as tf
import numpy


dataset_X = read_csv(r'./YuChen/train_data_x_v2.csv', delimiter=",", header=None)
print(dataset_X.head(5))
values_X = dataset_X.values
dataset_Y = read_csv(r'./YuChen/train_data_y_v2.csv', delimiter=",", header=None)
print(dataset_Y.head(5))
values_Y = dataset_Y.values
train_Y = values_Y.reshape((values_Y.shape[0]))
print(values_X.shape)
print(train_Y.shape)

labelFourAmount = 0
for i in range(len(train_Y)):
    if train_Y[i] == 4:
        labelFourAmount += 1

print(labelFourAmount)

train_X = values_X.reshape((values_X.shape[0], values_X.shape[1], -1))

model = tf.contrib.keras.models.load_model(r'./YuChen/300Ev3.h5')

predict = model.predict(train_X)
predict_Label = []

for pre in predict:
    maximum = 0
    for index in range(len(pre)):
        if pre[maximum] < pre[index]:
            maximum = index
    predict_Label.append(maximum)
# print(len(predict_Label))

sum = 0
for i in range(len(train_Y)):
    if predict_Label[i] != train_Y[i]:
        print(i)
        print(predict[i])
        sum += 1
        print('correct answer: ', train_Y[i])
        print('wrong answer: ', predict_Label[i])
        print('==============================')
print(sum)
