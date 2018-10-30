from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from LSTM_data import RunData
import numpy as np

instrument = ["EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_NZD","EUR_USD"]
granularity = 'H1'
candleCount = 24

# Data
model = load_model('Models/EUR_USD_time_series_LSTM_test.h5')
data = RunData(instrument, candleCount, granularity)
print(data)
data = data.reshape(1,data.shape[0],data.shape[2])
data_shape = data.shape

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform((data).reshape(-1,data.shape[1]))
data = data.reshape(data_shape)

# Predict EUR_USD
prediction = model.predict(data)
print(prediction)