import tensorflow as tf
import keras 

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

import time
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import numpy as np
from numpy import newaxis
import pandas as pd
from matplotlib import pyplot as plt

n=20
def load_data(n):
	df = pd.read_csv("Data/EUR_Keras.csv")
	chunked_df = [df[i:i+n].values.tolist() for i in range(0,df.shape[0],n)]
	return np.array(chunked_df[:-1])

def result_data(n):
	df = pd.read_csv("Data/EUR_Keras.csv")
	df[['close']]
	chunked_df = [df[i:i+n].values.tolist() for i in range(0,df.shape[0],n)]
	chunked_array = np.array(chunked_df)
	result = []
	for i in chunked_array:
		result.append(i[-1][-1])
	return np.array(result[1:])

x_train = load_data(n)
y_train = result_data(n).reshape(249)

print(x_train.shape, y_train.shape)
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform((x_train).reshape(-1,119520))
x_train = x_train.reshape(249,20,24)

model = Sequential()
model.add(LSTM(input_shape=(20,24), return_sequences=True, units=50))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Activation('linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop',)
history = model.fit(x_train, y_train, batch_size=10, epochs=20)

#history = model.fit(X, Y, epochs=100, validation_data=(valX, valY))
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
'''
def load_test_data():
	df = pd.read_csv("C:/Users/Kurt/Desktop/Python_CSV_Data/CSVData/testrun.csv")
	x = df.values.tolist()
	x = np.array(x)
	return x.reshape(1,100,24)

#predict = predict_sequences_multiple(model, load_test_data(n),99,99)
#plot_results_multiple(predict,0.1,99)
prediction = model.predict(load_test_data())
print(prediction)
'''