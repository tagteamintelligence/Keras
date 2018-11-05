from LSTM_data import Data 
import pandas as pd 
# Data(instrument, candleCount, granularity)
instrument = "EUR_USD"
granularity = 'H1'
candleCount = 1000

df = Data(instrument, candleCount, granularity)
df = pd.DataFrame(df)
df.columns = ['high','low','close']
rolling_list = [10,20,30,40,50,60]

def labeler(df, window=10):
	df['label_%d' % window] = 0
	for i in range(len(df['close'])):
		i_max = df['close'][window if (i-window) < window else i-window:i+window].max()
		if float(df['close'][i]) == float(i_max):
			df['label_%d' % window].loc[i] = 1
	return df
		
for windows in rolling_list:
	labeler(df, windows)

print(df)