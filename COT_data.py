from LSTM_data import Data 
import pandas as pd 

# instrument = "EUR_USD"
# granularity = 'H1'
# candleCount = 200

# df = pd.DataFrame(Data(instrument, candleCount, granularity, time=True))
# df.columns = ['time','high','low','close']
# df['time'] = pd.to_datetime(df['time'])
# df['time'] = pd.DatetimeIndex(df['time']).normalize()

def CAD_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	CAD_df = COT_df.loc[COT_df.name == 'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	CAD_df = CAD_df.set_index(CAD_df['date'])
	CAD_df = CAD_df.drop(columns=['name'])
	return CAD_df

def CHF_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	CHF_df = COT_df.loc[COT_df.name == 'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE']
	CHF_df = CHF_df.set_index(CHF_df['date'])
	CHF_df = CHF_df.drop(columns=['name'])
	return CHF_df

def GBP_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	GBP_df = COT_df.loc[COT_df.name == 'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE']
	GBP_df = GBP_df.set_index(GBP_df['date'])
	GBP_df = GBP_df.drop(columns=['name'])
	return GBP_df

def JPY_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	JPY_df = COT_df.loc[COT_df.name == 'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE']
	JPY_df = JPY_df.set_index(JPY_df['date'])
	JPY_df = JPY_df.drop(columns=['name'])
	return JPY_df

def EUR_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	EUR_df = COT_df.loc[COT_df.name == 'EURO FX - CHICAGO MERCANTILE EXCHANGE']
	EUR_df = EUR_df.set_index(EUR_df['date'])
	EUR_df = EUR_df.drop(columns=['name'])
	return EUR_df

def AUD_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	AUD_df = COT_df.loc[COT_df.name == 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	AUD_df = AUD_df.set_index(AUD_df['date'])
	AUD_df = AUD_df.drop(columns=['name'])
	return AUD_df

def NZD_COT():
	COT_df = pd.read_csv('COT_file/COT.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	NZD_df = COT_df.loc[COT_df.name == 'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	NZD_df = NZD_df.set_index(NZD_df['date'])
	NZD_df = NZD_df.drop(columns=['name','date'])
	return NZD_df

print(NZD_COT())



