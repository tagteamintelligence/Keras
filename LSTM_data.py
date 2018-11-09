import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from COT_data import Selection
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
config.read('../oanda_2018/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

def Data(instrument, candleCount, granularity, ma_list=False):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}
	candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
	candleData = client.request(candleData)
	candleData = candleData ['candles']
	df = pd.DataFrame([[str(x['time']) for x in candleData],
					   [str(x['mid']['h']) for x in candleData],
	    			   [str(x['mid']['l']) for x in candleData],
	    			   [str(x['mid']['c']) for x in candleData]]).T
	df.columns = ['time','high','low','close']
	df['time'] = pd.to_datetime(df['time'])
	if ma_list != False:
		df = IndicatorData(df, instrument, ma_list=ma_list)
	df = np.array(df.values)
	return df

def RunData(instrument, candleCount, granularity, time_series=1, close_only=False, time=False, ma_list=False):
	if time_series == 1:
		candleCount = candleCount+1
	shift = 0
	main_frame_len = candleCount-(max(ma_list)-1)
	main_frame = np.empty((main_frame_len,0))
	for i in range(len(instrument)):
		data = Data(instrument[i], candleCount, granularity, ma_list=ma_list)
		main_frame = np.append(main_frame, data, 1)
	main_frame = main_frame.reshape(1,main_frame.shape[0],main_frame.shape[1])
	
	if close_only == True:
		if time == True:
			section = main_frame[0,:,3]
		elif time == False:
			section = main_frame[0,:,2]
		section = section.reshape(section.shape[0],1)
		return section

	elif close_only == False: 
		big_main_frame = []
		for x in range(main_frame_len-time_series):
			section = main_frame[0,x:x+time_series,:]
			section = section.reshape(section.shape[0],section.shape[1])
			big_main_frame.append(section.tolist())
			shift += 1
		return np.array(big_main_frame)

def IndicatorData(df, instrument, ma_list=[], cot=True):
	for i in range(len(ma_list)):
		df['MA{}'.format(ma_list[i])] = df['close'].rolling(ma_list[i]).mean()

	if cot == True:
		code = [instrument[0:3],instrument[4:7]]
		if code[0] != 'USD':
			cot_history = Selection(code[0])
		if code[1] != 'USD':
			cot_history = Selection(code[1])

	df = df.set_index(df['time'])
	df = df.drop(columns=['time'])
	df = pd.concat([df, cot_history], axis=1, sort=False)
	df[['long','short']] = df[['long','short']].fillna(method='ffill')

	df = df.dropna()
	return df

def BollingerBand(data, window, num_std):
	df = pd.DataFrame(data)
	df.columns = ['close']
	roll_mean = df['close'].rolling(24).mean()
	roll_dev = df['close'].rolling(window).std()

	df['upper_band'] = roll_mean + (roll_dev*num_std)
	df['lower_band'] = roll_mean - (roll_dev*num_std)
	return df


# instrument = ["EUR_USD"]
# candleCount = 5000
# granularity = "H1"
# time_series = 3

# data = RunData(instrument, candleCount, granularity, time_series=time_series, close_only=True, ma_list=[5,10,20,50,100,200,250])
# print(data)