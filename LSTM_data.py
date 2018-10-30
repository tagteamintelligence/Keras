import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
config.read('../oanda_2018/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

def Data(instrument, candleCount, granularity):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}
	candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
	candleData = client.request(candleData)
	candleData = candleData ['candles']
	df = pd.DataFrame([[str(x['mid']['h']) for x in candleData],
	    			   [str(x['mid']['l']) for x in candleData],
	    			   [str(x['mid']['c']) for x in candleData]]).T
	df.columns = ['high','low','close']
	df = np.array(df.values)
	return df

def RunData(instrument, candleCount, granularity, time_series, close_only=False):
	shift = 0	
	main_frame = np.empty((candleCount,0))
	for i in range(len(instrument)):
		data = Data(instrument[i], candleCount, granularity)
		main_frame = np.append(main_frame, data, 1)
	main_frame = main_frame.reshape(1,main_frame.shape[0],main_frame.shape[1])
	
	if close_only == True:
		big_main_frame = np.empty((0,1,len(instrument)))
		#for x in range(candleCount-time_series):
		section = main_frame[0,:,2]
		section = section.reshape(section.shape[0],1)
		return section
	
	elif close_only == False:
		big_main_frame = np.empty((0,time_series,len(instrument)*3))
		for x in range(candleCount-time_series):
			section = main_frame[0,x:x+time_series,:]
			section = section.reshape(1,section.shape[0],section.shape[1])
			big_main_frame = np.append(big_main_frame, section, 0)
			shift += 1
			print('Data: '+str(shift)+'/'+str(candleCount-time_series))
		return big_main_frame


