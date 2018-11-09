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
from LSTM_data import RunData
import time

### USER INPUT ###
main_pair = ["EUR_USD"]#,"USD_JPY","GBP_USD","AUD_USD","NZD_USD","USD_CHF","USD_CAD"]
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
epochs = 5
batch_size = 25
candleCount = 4800
ma_list = [5,10,20,50,100,200,250]

print('CandleCount:',candleCount,'of MAX 5000')

plt.figure(num='AI Model')
for x in range(len(main_pair)):
	instrument = [i for i in all_instruments if main_pair[x][0:3] in i]
	instrument = instrument+[i for i in all_instruments if main_pair[x][4:7] in i]

	# x_train
	x_train = None
	while x_train is None:
		try:
			x_train = RunData([main_pair[x]], candleCount, granularity, time_series=time_series, ma_list=ma_list)
		except:
			pass
			print('x_train Oanda Error')
			time.sleep(10)
	print('Done Loading '+main_pair[x]+' x_train')
	time.sleep(2)
	# y_train
	y_train = None
	while y_train is None:
		try:
			y_train = RunData([main_pair[x]], candleCount, granularity, time_series=time_series, close_only=True)
		except:
			pass
			print('y_train Oanda Error')
			time.sleep(10)
	print('Done Loading '+main_pair[x]+' y_train')

	# Batch Sizing
	if ma_list != None:
		x_train = x_train[:-look_forward]
		y_train = y_train[time_series+look_forward+(max(ma_list)-1):]
		x_train_shape = x_train.shape
	x_train = x_train[:-look_forward]
	y_train = y_train[time_series+look_forward:]
	x_train_shape = x_train.shape

	# Scale
	scaler = MinMaxScaler(feature_range=(0,1))
	x_train = scaler.fit_transform((x_train.astype(float)).reshape(x_train_shape[0]*x_train_shape[1],x_train_shape[2]))
	x_train = x_train.reshape(x_train_shape)

	# Create LSTM Model
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2])))
	model.add(Dropout(0.2))
	model.add(LSTM(50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(50))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation="linear"))
	model.compile(loss='mean_squared_error', optimizer='adam')
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=2)
	model.save('Models/'+main_pair[x]+'/'+granularity+'/'+str(time_series)+'_'+str(look_forward)+'_LSTM_MA.h5')
	print(main_pair[x], 'Model Saved')

	plt.subplot(4,2,x+1)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(main_pair[x]+' model train vs validation loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	
plt.show()