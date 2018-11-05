from LSTM_data import Data 
import pandas as pd 
# Data(instrument, candleCount, granularity)
instrument = "EUR_USD"
granularity = 'H1'
candleCount = 1000

df = Data(instrument, candleCount, granularity)
df = pd.DataFrame(df)
df.columns = ['high','low','close']
rolling_list = [i for i in range(5,50,5)]

def labeler(df, window=10):
	df['label_%d' % window] = 0
	for i in range(len(df['close'])):
		i_max = df['close'][window if (i-window) < window else i-window:i+window].max()
		i_min = df['close'][window if (i-window) < window else i-window:i+window].min()
		if float(df['close'][i]) == float(i_max):
			df['label_%d' % window].loc[i] = 1
		if float(df['close'][i]) == float(i_min):
			df['label_%d' % window].loc[i] = 1
	return df
		
def label_maker(rolling_list):
	for windows in rolling_list:
		labeler(df, windows)


label_maker(rolling_list)
print(df)