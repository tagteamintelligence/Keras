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
import pandas as pd
n=100
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
y_train = result_data(n).reshape(49,1)

print(x_train.shape, y_train.shape)
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform((x_train).reshape(-1,117600))
x_train = x_train.reshape(49,100,24)

model = Sequential()
model.add(LSTM(input_shape=(100,24), return_sequences=True, units=50))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Activation('linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop',)
model.fit(x_train, y_train, batch_size=100, epochs=50)


