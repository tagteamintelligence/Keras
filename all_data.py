import keras 
# Model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Bidirectional
from keras.optimizers import Adam
# Data
import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
pd.set_option('display.max_columns',10)
pd.set_option('display.width',None)

class Indicators:
	def __init__(self, df):
		self.df = df

	def SMA(self, ma_list, label='close'):
		df = self.df
		for i in range(len(ma_list)):
			df['SMA_{}'.format(ma_list[i])] = df[label].rolling(ma_list[i]).mean()
		return df

	def COT(self, instrument):
		from COT_data import COT
		first, second = instrument[0:3], instrument[4:7]
		df = self.df
		df = df.set_index(df['time'])
		df = df.drop(columns=['time'])
		if first == 'CAD' or second == 'CAD':
			df = pd.concat([df,COT.CAD()], axis=1, sort=False)
		if first == 'CHF' or second == 'CHF':
			df = pd.concat([df,COT.CHF()], axis=1, sort=False)
		if first == 'GBP' or second == 'GBP':
			df = pd.concat([df,COT.GBP()], axis=1, sort=False)
		if first == 'JPY' or second == 'JPY':
			df = pd.concat([df,COT.JPY()], axis=1, sort=False)
		if first == 'EUR' or second == 'EUR':
			df = pd.concat([df,COT.EUR()], axis=1, sort=False)
		if first == 'AUD' or second == 'AUD':
			df = pd.concat([df,COT.AUD()], axis=1, sort=False)
		if first == 'NZD' or second == 'NZD':
			df = pd.concat([df,COT.NZD()], axis=1, sort=False)
		if first != 'USD' and second != 'USD':
			df.iloc[:,-4:] = df.iloc[:,-4:].fillna(method='ffill')
			return df
		df[['long','short']] = df[['long','short']].fillna(method='ffill')
		df = df.dropna()
		return df


class Data:
	def __init__(self, instrument, granularity, candleCount):
		self.instrument = instrument
		self.granularity = granularity
		self.candleCount = candleCount
		config = configparser.ConfigParser()
		config.read('../oanda_2018/config/config_v20.ini')
		accountID = config['oanda']['account_id']
		access_token = config['oanda']['api_key']
		client = oandapyV20.API(access_token=access_token)
		params = {"count": candleCount, "granularity": granularity}
		candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
		candleData = client.request(candleData)
		self.candleData = candleData ['candles']

	def OHLC(self, time=False):
		df = pd.DataFrame([[x['time'] for x in self.candleData],
						   [x['mid']['o'] for x in self.candleData],
						   [x['mid']['h'] for x in self.candleData],
						   [x['mid']['l'] for x in self.candleData],
						   [x['mid']['c'] for x in self.candleData]]).T
		df.columns = ['time','open','high','low','close']
		if time == True:
			df['time'] = pd.to_datetime(df['time'])
			if self.granularity == 'D':
				df['time'] = [x.replace(hour=14) for x in df['time']]
				return df
			return df
		return df[['open','high','low','close']]

	def HLC(self, time=False):
		df = pd.DataFrame([[x['time'] for x in self.candleData],
						   [x['mid']['h'] for x in self.candleData],
						   [x['mid']['l'] for x in self.candleData],
						   [x['mid']['c'] for x in self.candleData]]).T
		df.columns = ['time','high','low','close']
		if time == True:
			df['time'] = pd.to_datetime(df['time'])
			if self.granularity == 'D':
				df['time'] = [x.replace(hour=14) for x in df['time']]
				return df
			return df
		return df[['high','low','close']]

	def HL(self, time=False):
		df = pd.DataFrame([[x['time'] for x in self.candleData],
						   [x['mid']['h'] for x in self.candleData],
						   [x['mid']['l'] for x in self.candleData]]).T
		df.columns = ['time','high','low']
		if time == True:
			df['time'] = pd.to_datetime(df['time'])
			if self.granularity == 'D':
				df['time'] = [x.replace(hour=14) for x in df['time']]
				return df
			return df
		return df[['high','low']]

	def Close(self, time=False):
		df = pd.DataFrame([[x['time'] for x in self.candleData],
						   [x['mid']['c'] for x in self.candleData]]).T
		df.columns = ['time','close']
		if time == True:
			df['time'] = pd.to_datetime(df['time'])
			if self.granularity == 'D':
				df['time'] = [x.replace(hour=14) for x in df['time']]
				return df
			return df
		return df[['close']]


class Percent: # Percentage candle data, for Train_Data()
	def __init__(self, df):
		self.df = df

	def HLC(self, time=False):
		df = self.df
		df['PCT_close'] = ((df['close'].astype(float) - df['open'].astype(float)) / df['close'].astype(float))*100
		df['PCT_high'] = ((df['close'].astype(float) - df['high'].astype(float)) / df['close'].astype(float))*100
		df['PCT_low'] = ((df['close'].astype(float) - df['low'].astype(float)) / df['close'].astype(float))*100
		if time == True:
			return df[['time','PCT_close','PCT_high','PCT_low']]
		return df[['PCT_close','PCT_high','PCT_low']]

	def Close(self, time=False):
		df = self.df
		df['PCT_close'] = ((df['close'].astype(float) - df['open'].astype(float)) / df['close'].astype(float))*100
		if time == True:
			return df[['time','PCT_close']]
		return df[['PCT_close']]

	def Range(self, look_forward):
		import numpy as np
		df = self.df
		df_list = []
		for x in range(self.df.shape[0]-look_forward):	
			section = ((df.loc[x+look_forward].astype(float) - df.loc[x].astype(float)) / df.loc[x+look_forward].astype(float))*100
			df_list.append(section.tolist())
		return np.array(df_list)

	def SMA(self, ma_list):
		df = self.df
		df = Indicators(df).SMA(ma_list)
		for x in ma_list:
			df['PCT_SMA_{}'.format(x)] = ((df['close'].astype(float) - df['SMA_{}'.format(x)].astype(float)) / df['close'].astype(float))*100
			df = df.drop(columns=['SMA_{}'.format(x)])
		return df


class Feature:
	def __init__(self, df):
		self.df = df

	def Clean(self):
		x1 = len(self.df)
		df = self.df.dropna()
		x2 = len(df)
		df = df.reset_index(drop=True)
		print('Values Cleaned:',x1-x2)
		return df

	def Square(self, label_list=['high','low']):
		df = self.df
		for x in label_list:
			df['square_{}'.format(x)] = df[x].astype(float)**2
		return df

	def Histogram(self, bins):
		df = self.df
		import matplotlib.pyplot as plt
		for i in df.columns.values:
			plt.title(i)
			plt.hist(df[i], density=True, bins=bins)
			plt.show()
		return df

	def Bins(self, bins, labels=False):
		df = self.df
		return pd.qcut(df, bins, labels=labels)


class Time_Series:
	def __init__(self, df, value):
		self.df = df
		self.value = value

	def Section(self): # Return numpy array
		import numpy as np
		df = self.df
		value = self.value
		df = np.array(df.values)
		df = df.reshape(1,df.shape[0],df.shape[1])
		df_list = []
		for x in range(len(df[0])-value):
			section = df[0,x:x+value,:]
			df_list.append(section.tolist())
		return np.array(df_list)

	def Chunk(self): # Return numpy array
		pass


class Train_Data: # Create data to be applied to Train()
	def __init__(self, df):
		self.data = df

	def SMA(self, ma_list):
		df = Percent(self.data).SMA(ma_list)
		df['PCT_close'] = Percent(df).Close()
		df['bin_PCT_close'] = Feature(df['PCT_close']).Bins(101)
		df = Feature(df).Clean()
		for x in ma_list:
			df['SMA_{}'.format(x)] = Feature(df['PCT_SMA_{}'.format(x)]).Bins(41)
			df = df.drop(columns=['PCT_SMA_{}'.format(x)])
		df = df.drop(columns=['open','high','low','close','PCT_close'])
		return df

	def SMA2(self, ma_list):
		df = Percent(self.data).SMA(ma_list)
		df['PCT_close'] = Percent(df).Close()
		df = Feature(df).Clean()
		df = df.drop(columns=['open','high','low','close'])
		return df


class Train: # Create a new set of training data
	def __init__(self, instrument, granularity, candleCount, time_series, look_forward):
		self.instrument = instrument
		self.granularity = granularity
		self.candleCount = candleCount
		self.look_forward = look_forward

	def X_Train_SMA(self, ma_list): # no loss gain
		data = None
		while data is None:
			try:
				data = Data(self.instrument, self.granularity, self.candleCount).OHLC()
			except:
				print('x_train Oanda Error')
				time.sleep(10)
		df = Train_Data(data).SMA2(ma_list)
		df = Time_Series(df, time_series).Section()
		self.df = df
		df = df[:-self.look_forward]
		x_train = Scale(df).Min_Max()
		print('x_train Done Loading')
		return x_train

	def Y_Train_PCT(self): # Range Percent
		data = None
		while data is None:
			try:
				data = Data(self.instrument, self.granularity, self.df.shape[0]).Close()
			except:
				print('y_train Oanda Error')
				time.sleep(10)
		y_train = Percent(data).Range(self.look_forward)
		print('y_train Done Loading')
		return y_train

	def X_Train_SMA_2(self, ma_list): # Not Done
		data = None
		while data is None:
			try:
				data = Data(self.instrument, self.granularity, self.candleCount).OHLC()
			except:
				print('x_train Oanda Error')
				time.sleep(10)
		df = Train_Data(data).SMA2(ma_list)
		df_2 = df.pow(2)
		df =  pd.concat([df,df_2], axis=1)
		df = Time_Series(df, time_series).Section()
		self.df = df
		df = df[:-self.look_forward]
		x_train = Scale(df).Min_Max()
		print('x_train Done Loading')
		return x_train

	def Y_Train_PCT_Bin(self): # Range Percent
		data = None
		while data is None:
			try:
				data = Data(self.instrument, self.granularity, self.df.shape[0]).Close()
			except:
				print('y_train Oanda Error')
				time.sleep(10)
		y_train = Percent(data).Range(self.look_forward)
		y_train = y_train.reshape(y_train.shape[0],)
		y_train = Feature(y_train).Bins(41)
		y_train = y_train.reshape(y_train.shape[0],1)
		print('y_train Done Loading')
		return y_train


class Scale:
	def __init__(self, x_train):
		self.x_train = x_train

	def Min_Max(self):
		from sklearn.preprocessing import MinMaxScaler
		x_train = self.x_train
		x_train_shape = x_train.shape
		scaler = MinMaxScaler(feature_range=(-1,1))
		x_train = scaler.fit_transform((x_train.astype(float)).reshape(x_train_shape[0]*x_train_shape[1],x_train_shape[2]))
		x_train = x_train.reshape(x_train_shape)
		return x_train


class Model:
	def __init__(self, x_train, y_train, instrument, granularity, time_series, look_forward, batch_size=10, epochs=10):
		self.x_train = x_train
		self.y_train = y_train

		self.instrument = instrument
		self.granularity = granularity
		self.time_series = time_series
		self.look_forward = look_forward
		self.batch_size = batch_size
		self.epochs = epochs
		self.x_train_shape = x_train.shape

	def Predict(self):
		from keras.models import load_model
		data = self.x_train[-1]
		data = data.reshape(1,data.shape[0],data.shape[1])
		model = load_model('Models/'+self.instrument+'/'+self.granularity+'/'+str(self.time_series)+'_'+str(self.look_forward)+'_LSTM_One.h5')
		prediction = model.predict(data)
		print(self.instrument+":", prediction)

	def LSTM_One(self):
		model = Sequential()
		model.add(LSTM(32, return_sequences=True, input_shape=(self.x_train_shape[1], self.x_train_shape[2])))
		model.add(SpatialDropout1D(0.4))
		model.add(LSTM(16, return_sequences=True))
		model.add(SpatialDropout1D(0.4))
		model.add(LSTM(8))
		model.add(Dense(1, activation="linear"))
		optimizer = Adam(lr=0.01)
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.3, verbose=2)
		model.save('Models/'+self.instrument+'/'+self.granularity+'/'+str(self.time_series)+'_'+str(self.look_forward)+'_LSTM_One.h5')
		print('Model Saved')
		return history


class Plot:
	def __init__(self, history, instrument):
		self.history = history
		self.instrument = instrument

	def Plot(self):
		import matplotlib.pyplot as plt
		history = self.history
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title(self.instrument+' model train vs validation loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper right')
		plt.show()
		

if __name__ == '__main__':
	instrument = 'EUR_USD'
	granularity = 'H1'
	candleCount = 4800
	time_series = 240
	look_forward = 24
	ma_list = [5,10,20,50,100,200,250]
	batch_size = 20
	epochs = 1000

	train = Train(instrument, granularity, candleCount, time_series, look_forward)
	x_train = train.X_Train_SMA_2(ma_list)
	y_train = train.Y_Train_PCT()
	print(x_train.shape)
	print(y_train.shape)

	# Model
	history = Model(x_train, y_train, instrument, granularity, time_series, look_forward, batch_size=batch_size, epochs=epochs).LSTM_One()
	Plot(history, instrument).Plot()
	predict = Model(x_train, y_train, instrument, granularity, time_series, look_forward, batch_size=batch_size, epochs=epochs).Predict()