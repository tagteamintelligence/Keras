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
from data_setup_for_LSTM import GetData

### USER INPUT ###
instrument = ["EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_NZD","EUR_USD"]
candleCount = 20
granularity = 'H1'

df = GetData(instrument, candleCount, granularity)
df_array = np.array(df.values.tolist())
x_train = df_array
y_train = df_array

print(x_train.shape, y_train.shape)
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform((x_train).reshape(-1,119520))
x_train = x_train.reshape(249,20,24)

exit()

model = Sequential()
model.add(LSTM(input_shape=(20,24), return_sequences=True, units=50))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Activation('linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_train, y_train, batch_size=10, epochs=20)
model.save('Models/EUR_20period_LSTM.h5')

history = model.fit(x_train, y_train, batch_size=10, epochs=20)
plt.plot(history.history['loss'])
plt.title('train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()