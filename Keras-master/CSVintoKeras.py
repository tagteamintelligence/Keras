import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import configparser
import time
import csv

# User Input
currencyPair = "EUR_CAD"
granularity = 'H4'
candleCount = 5000

# Oanda Login
config = configparser.ConfigParser()
config.read('../config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']
client = oandapyV20.API(access_token=access_token)

# Data to CSV
params = {
	"count": candleCount,
	"granularity": granularity
}
candleData = instruments.InstrumentsCandles(instrument= currencyPair,params= params)
hourly = client.request(candleData)
hourly = hourly['candles']
df = pd.DataFrame([
    [float(x['mid']['o']) for x in hourly],
    [float(x['mid']['h']) for x in hourly],
    [float(x['mid']['l']) for x in hourly],
    [float(x['mid']['c']) for x in hourly]]).T 
df.columns = ('open', 'high', 'low', 'close')
df['PCT_change'] = (df['close'] - df['open']) / df['open']*100
df['result'] = [1 if x > 0 else 0 for x in df['PCT_change']]
df['result'] = df['result'].shift(-1)

df = df[['result', 'open', 'high', 'low', 'close']]
df = df.drop(df.index[-1])
print(df)

# Write to Data Folder
df.to_csv("Data/Keras/"+currencyPair+"_"+granularity+".csv", sep=',', encoding='utf-8', index=False)