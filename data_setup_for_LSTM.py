import numpy as np 
import pandas as pd 

example = [[[1,2,3],[4,5,6]],[[2,3,4],[5,6,7]],[[3,4,5],[6,7,8]]]
print(pd.DataFrame(example))

import csv
import pandas as pd
import configparser
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import time

config = configparser.ConfigParser()
config.read('C:/Users/Kurt/Desktop/pythonCode/Python_CSV_Data/config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

instrument = ["EUR_USD", "EUR_GBP"]

candleCount = 3
params = {
    "count": candleCount,
    "granularity": 'H1'
}
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

bigFrame = pd.DataFrame()
for x in range(len(instrument)):
    bigFrame = pd.concat([bigFrame, Run(instrument[x], params)], axis=1)
bigFrame.columns = instrument
print(bigFrame)