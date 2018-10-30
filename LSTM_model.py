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
from LSTM_data_new import RunData

### USER INPUT ###
main_pair = ["EUR_USD"]
instrument = ["EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_NZD","EUR_USD"]
granularity = 'H1'
time_series = 24
nCycle = 200
epochs = 5
candleCount = time_series*nCycle
print('CandleCount:',candleCount,'of MAX 5000')

# Array Data
x_train = RunData(instrument, candleCount, granularity, time_series)
y_train = RunData(main_pair, candleCount, granularity, time_series, close_only=True)

# Batch Sizing
x_train = x_train[:]
y_train = y_train[time_series:]
x_train_shape = x_train.shape

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform((x_train).reshape(-1,x_train.shape[1]))
x_train = x_train.reshape(x_train_shape)

# Create LSTM Model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units= 100, return_sequences=False))
model.add(Dense(activation="linear", units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, batch_size=1, epochs=epochs, verbose=1)
#model.save('Models/'+main_pair+'_'+'time_series'+'_LSTM_test.h5')

plt.plot(history.history['loss'])
plt.title('train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()