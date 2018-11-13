import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd

class Indicators:
	def __init__(self, df):
		self.df = df

	def Clean(self):
		x1 = len(self.df)
		df = self.df.dropna()
		x2 = len(df)
		print('Values Cleaned:',x1-x2)
		return df

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
						   [x['mid']['c'] for x in self.candleData],]).T
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
						   [x['mid']['c'] for x in self.candleData],]).T
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
						   [x['mid']['l'] for x in self.candleData],]).T
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
						   [x['mid']['c'] for x in self.candleData],]).T
		df.columns = ['time','close']
		if time == True:
			df['time'] = pd.to_datetime(df['time'])
			if self.granularity == 'D':
				df['time'] = [x.replace(hour=14) for x in df['time']]
				return df
			return df
		return df[['close']]

	def Percent(self, time=False):
		if time == True:
			df = Data(self.instrument, self.granularity, self.candleCount).OHLC(time=True)
		if time == False:
			df = Data(self.instrument, self.granularity, self.candleCount).OHLC()
		df['PCT_close'] = ((df['open'].astype(float) - df['close'].astype(float)) / df['open'].astype(float))*100
		df['PCT_high'] = ((df['open'].astype(float) - df['high'].astype(float)) / df['open'].astype(float))*100
		df['PCT_low'] = ((df['open'].astype(float) - df['low'].astype(float)) / df['open'].astype(float))*100
		if time == True:
			return df[['time','PCT_close','PCT_high','PCT_low']]
		return df[['PCT_close','PCT_high','PCT_low']]


class Feature:
	def __init__(self, df):
		self.df = df

	def Square(self, label_list=['high','low']):
		df = self.df
		for x in label_list:
			df['square_{}'.format(x)] = df[x].astype(float)**2
			n += 1
		return df

	def Histogram(self, bins):
		df = self.df
		import matplotlib.pyplot as plt
		for i in df.columns.values:
			plt.title(i)
			plt.hist(df[i], density=True, bins=bins)
			plt.show()
		return df

	def Bins(self, label, bins, labels=False):
		df = self.df
		df['bin_{}'.format(label)] = pd.cut(df[label], bins, labels=labels)
		return df


class TimeSeries:
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


if __name__ == '__main__':
	instrument = 'EUR_USD'
	granularity = 'H1'
	candleCount = 5000

	df = Data(instrument, granularity, candleCount).Percent()
	df = Feature(df).Bins('PCT_close',100)

	print(df)




