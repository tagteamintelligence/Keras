import keras 
# Model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
# Support
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
from LSTM_data import GetData
from LSTM_data import GetResultData

### USER INPUT ###
main_pair = "EUR_USD"
instrument = ["EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_NZD","EUR_USD"]
granularity = 'H1'
batch_size = 24
nCycle = 200
epochs = 20
candleCount = batch_size*nCycle

# Array Data
x_train = np.array(GetData(instrument, candleCount, granularity).values)
y_train = np.array(GetResultData(main_pair, candleCount, granularity).values)

# Batch Sizing
x_train = x_train[:-batch_size]
y_train = y_train[batch_size:]

# Reshapeing
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_train_shape = x_train.shape

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform((x_train).reshape(-1,x_train.shape[1]))
x_train = x_train.reshape(x_train_shape)

print(x_train.shape, y_train.shape)
# create LSTM
model = Sequential()
model.add(LSTM(units= 100, input_shape=(x_train_shape[1], x_train_shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
model.save('Models/EUR_24period_LSTM.h5')

plt.plot(history.history['loss'])
plt.title('train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()