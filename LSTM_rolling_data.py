from LSTM_data import Data 
import pandas as pd
import numpy as np 
# Data(instrument, candleCount, granularity)


def labeler(df, window=10):
	df['label_%d' % window] = "sit"
	for i in range(len(df['close'])):
		i_max = df['close'][window if (i-window) < window else i-window:i+window].max()
		i_min = df['close'][window if (i-window) < window else i-window:i+window].min()
		if float(df['close'][i]) == float(i_max):
			df['label_%d' % window].loc[i] = "long"
		if float(df['close'][i]) == float(i_min):
			df['label_%d' % window].loc[i] = "short"
	section = np.array(df['label_%d' % window].values)
	section = section.reshape(section.shape[0],1)
	return section
		
def label_maker(instrument, candleCount, granularity):
	df = Data(instrument, candleCount, granularity)
	df = pd.DataFrame(df)
	df.columns = ['high','low','close']
	return labeler(df)

def array_length_check(long_df, short_df, sit_df):
	x = min(len(long_df), len(short_df), len(sit_df))
	return x

#def cycle_rows():

def dataset(X_train_np, y_train_np):
	dfX = pd.DataFrame(X_train_np.tolist())
	#print(dfX)
	dfy = pd.DataFrame(y_train_np.tolist())
	#print(dfy)
	longX = dfX.loc[dfy[0] == 'long']
	longy = dfy.loc[dfy[0] == 'long']

	shortX = dfX.loc[dfy[0] == 'short']
	shorty = dfy.loc[dfy[0] == 'short']

	sitX = dfX.loc[dfy[0] == 'sit']
	sity = dfy.loc[dfy[0] == 'sit']

	index = array_length_check(longX, shortX, sitX)

	X_train = longX[:index].append(shortX[:index])
	X_train = X_train.append(sitX[:index])
	print(X_train)
	y_train = longy[:index].append(shorty[:index])
	y_train = y_train.append(sity[:index])
	print(y_train)

	#print(long_list, short_list.shape, sit_list.shape)
	return
