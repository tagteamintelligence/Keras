import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
import numpy as np 

config = configparser.ConfigParser()
config.read('C:/Users/Kurt/Desktop/pythonCode/oanda_2018/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

def Run(instrument, params):
	candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
	candleData = client.request(candleData)
	candleData = candleData ['candles']
	df = pd.DataFrame([[str(x['mid']['o']) for x in candleData],
		[str(x['mid']['h']) for x in candleData],
	    [str(x['mid']['l']) for x in candleData],
	    [str(x['mid']['c']) for x in candleData],
	    [str(x['volume']) for x in candleData]]).T
	df.columns = ['open','high','low','close','volume']
	return df[['open','high','low','close']]

def GetData(instrument, candleCount, granularity):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}
	bigFrame = pd.DataFrame()
	for x in range(len(instrument)):
	    bigFrame = pd.concat([bigFrame, Run(instrument[x], params)], axis=1)
	return bigFrame

def GetResultData(instrument, candleCount, granularity):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}
	df = Run(instrument, params)[['close']]
	return df