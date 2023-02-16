from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from numpy import concatenate
import csv

dataset = read_csv(r'./vali_LOC.csv', delimiter=",", header=None)

print(dataset)

values = dataset.values

values_X, values_Y = values[:, :-1], values[:, -1]

# ensure all data in float
values_X = values_X.reshape((values_X.shape[0], values_X.shape[1], -1))
print(values_X.shape)

model = tf.contrib.keras.models.load_model(r'./model_save/model_LOC_4L_300E.h5')

predict = model.predict(values_X)
predict_Label = []
for pre in predict:
    maximum = 0
    for i in range(len(pre)):
        if pre[maximum] < pre[i]:
            maximum = i
    predict_Label.append(maximum)

print(predict_Label)



