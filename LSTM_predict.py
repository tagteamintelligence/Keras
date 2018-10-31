from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from LSTM_data import RunData
import numpy as np

main_pair = ["USD_JPY"] #EUR_USD USD_JPY GBP_USD AUD_USD NZD_USD USD_CHF USD_CAD
all_instruments = ["AUD_CAD","AUD_CHF","AUD_JPY","AUD_NZD","AUD_SGD","AUD_USD",
				   "CAD_CHF","CAD_JPY","CAD_SGD",
			  	   "CHF_JPY",
			  	   "EUR_AUD","EUR_CAD","EUR_CHF","EUR_GBP","EUR_JPY","EUR_NZD","EUR_SGD","EUR_USD",
			  	   "GBP_AUD","GBP_CAD","GBP_CHF","GBP_JPY","GBP_NZD","GBP_SGD","GBP_USD",
			  	   "NZD_CAD","NZD_CHF","NZD_JPY","NZD_SGD","NZD_USD",
			  	   "SGD_CHF","SGD_JPY",
			  	   "TRY_JPY",
			  	   "USD_CAD","USD_CHF","USD_CNH","USD_HKD","USD_JPY","USD_SGD","USD_THB",
			  	   "ZAR_JPY"]
instrument = [i for i in all_instruments if main_pair[0][0:3] in i]
instrument = instrument+[i for i in all_instruments if main_pair[0][4:7] in i]
granularity = 'H1'
time_series = 24
nCycle = 200
candleCount = time_series*nCycle

# Data
model = load_model('Models/'+main_pair[0]+'_'+granularity+'_time_series_LSTM.h5')
print('Model Loaded with CandleCount:',candleCount,'of MAX 5000')
data = RunData(instrument, candleCount, granularity)
data_shape = data.shape

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform((data).reshape(data_shape[0]*data_shape[1],data_shape[2]))

data = data.reshape(data_shape)
data = data[-time_series:]
data = data.reshape(1,time_series,data_shape[2])
# Predict EUR_USD
prediction = model.predict(data)
print(prediction)