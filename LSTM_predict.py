from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from LSTM_data import RunData
from matplotlib import pyplot as plt
from LSTM_data import BollingerBand
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
time_series = 240
look_forward = 24
candleCount = 4800

plt.figure(num='AI prediction')
for i in range(len(main_pair)):
	# Data
	instrument = [x for x in all_instruments if main_pair[i][0:3] in x]
	instrument = instrument+[x for x in all_instruments if main_pair[i][4:7] in x]
	model = load_model('Models/'+main_pair[i]+'/'+granularity+'/'+str(time_series)+'_'+str(look_forward)+'_LSTM.h5')
	print(str(i+1)+'/'+str(len(main_pair))+' Loaded with CandleCount:',candleCount,'of MAX 5000')
	data = RunData(instrument, candleCount, granularity)
	data_shape = data.shape

	scaler = MinMaxScaler(feature_range=(0,1))
	data = scaler.fit_transform((data.astype(float)).reshape(data_shape[0]*data_shape[1],data_shape[2]))

	data = data.reshape(data_shape)
	data = data[-time_series:]
	data = data.reshape(1,time_series,data_shape[2])
	# Predict EUR_USD
	prediction = model.predict(data)
	print(main_pair[i]+":", prediction)

	plot_value_shape = (time_series*5)+1
	data = RunData([main_pair[i]], time_series*5, granularity, close_only=True).reshape(plot_value_shape)
	plot_value = data.tolist()

	plt.subplot(4,2,i+1)
	plt.title(main_pair[i])
	plt.plot([float(i) for i in plot_value], color='blue')
	plt.plot((time_series*5)+look_forward, float(prediction[0]), marker='o', markersize=5, color="red")
	plt.ylabel('Close Price')
	plt.show(block=False)

plt.show()