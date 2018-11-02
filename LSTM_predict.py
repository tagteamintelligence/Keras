from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from LSTM_data import RunData
from matplotlib import pyplot as plt
import numpy as np

main_pair = ["EUR_USD","USD_JPY","GBP_USD","AUD_USD","NZD_USD","USD_CHF","USD_CAD"]
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

granularity = 'H1'
time_series = 120
nCycle = 40
candleCount = time_series*nCycle

for i in range(len(main_pair)):
	# Data
	instrument = [x for x in all_instruments if main_pair[i][0:3] in x]
	instrument = instrument+[x for x in all_instruments if main_pair[i][4:7] in x]
	model = load_model('Models/'+main_pair[i]+'_'+granularity+'_time_series_'+str(time_series)+'_LSTM.h5')
	print('Model Loaded with CandleCount:',candleCount,'of MAX 5000')
	data = RunData(instrument, candleCount, granularity)
	data_shape = data.shape

	scaler = MinMaxScaler(feature_range=(0,1))
	data = scaler.fit_transform((data).reshape(data_shape[0]*data_shape[1],data_shape[2]))

	data = data.reshape(data_shape)
	data = data[-time_series:]
	data = data.reshape(1,time_series,data_shape[2])
	print(data.shape)
	# Predict EUR_USD
	prediction = model.predict(data)
	print(prediction)

	plot_value_shape = (time_series*5)+1
	plot_value = RunData([main_pair[i]], time_series*5, granularity, close_only=True).reshape(plot_value_shape).tolist()
	plt.figure(num=main_pair[i])
	plt.plot([float(i) for i in plot_value])
	plt.plot((time_series*5)+time_series+1, float(prediction[0]), marker='o', markersize=5, color="red")
	plt.title(main_pair[i])
	plt.xlabel('Candle Count')
	plt.ylabel('Close Price')
	plt.show(block=False)

plt.show()