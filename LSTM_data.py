import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
config.read('../../oanda_2018/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

def Data(instrument, candleCount, granularity, shift, time_series, close_only=False):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}
	candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
	candleData = client.request(candleData)
	candleData = candleData ['candles']
	df = pd.DataFrame([[str(x['mid']['o']) for x in candleData],
		[str(x['mid']['h']) for x in candleData],
	    [str(x['mid']['l']) for x in candleData],
	    [str(x['mid']['c']) for x in candleData],
	    [str(x['volume']) for x in candleData]]).T
	df.columns = ['open','high','low','close','volume']
	out = pd.DataFrame([df['high'][shift:shift+time_series],
						df['low'][shift:shift+time_series],
						df['close'][shift:shift+time_series]])
	if close_only == True:
		out = pd.DataFrame([df['close'][shift:shift+1]])
		out = np.array(out.values)
		out = np.transpose(out)
		out = out.reshape(1,1,1)
		return out
	out = np.array(out.values)
	out = np.transpose(out)
	out = out.reshape(1,time_series,3)
	return out

def RunData(instrument, candleCount, granularity, time_series, close_only=False):
	if close_only == True:
		big_main_frame = np.empty((0,1,len(instrument)*1))
		main_frame = np.empty((1,1,0))
	else:
		big_main_frame = np.empty((0,time_series,len(instrument)*3))
		main_frame = np.empty((1,time_series,0))
	shift = 0
	for x in range(candleCount-time_series):
		for i in range(len(instrument)):
			data = Data(instrument[i], candleCount, granularity, shift, time_series, close_only)
			main_frame = np.append(main_frame, data, 2)
		shift += 1
		big_main_frame = np.append(big_main_frame, main_frame, 0)
		if close_only == True:
			main_frame = np.empty((1,1,0))
		else:
			main_frame = np.empty((1,time_series,0))
		print('Data: '+str(shift)+'/'+str(candleCount-time_series))
	return big_main_frame