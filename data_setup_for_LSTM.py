import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import pandas as pd
import numpy as np 

config = configparser.ConfigParser()
config.read('C:/Users/Kurt/Desktop/pythonCode/Python_CSV_Data/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

def Run(instrument, params):
	candleData = instruments.InstrumentsCandles(instrument=instrument, params=params)
	candleData = client.request(candleData)
	candleData = candleData ['candles']
	#print(candleData)
	df = pd.DataFrame()
	for i in candleData:
		df = pd.concat([df,pd.Series([[float(i['mid']['o']),float(i['mid']['h']),float(i['mid']['l']),float(i['mid']['c'])]])])
		df = df.reset_index(drop=True)
	return df

def GetData(instrument, candleCount, granularity):
	params = {
	    "count": candleCount,
	    "granularity": granularity
	}

	bigFrame = pd.DataFrame()
	for x in range(len(instrument)):
	    bigFrame = pd.concat([bigFrame, Run(instrument[x], params)], axis=1)
	bigFrame = bigFrame.T
	bigFrame.index = instrument
	return bigFrame