from keras.models import load_model
from LSTM_data import GetData
from sklearn.preprocessing import MinMaxScaler
import numpy as np

instrument = ["EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_NZD","EUR_USD"]
granularity = 'H1'
batch_size = 24

# Data
model = load_model('Models/EUR_24period_LSTM.h5')
data = np.array(GetData(instrument, batch_size, granularity).values)
data = data.reshape(data.shape[0],1,data.shape[1])
data_shape = data.shape

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform((data).reshape(-1,data.shape[1]))
data = data.reshape(data_shape)

# Predict EUR_USD
prediction = model.predict(data)
print(prediction)