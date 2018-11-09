import pandas as pd 

def CAD_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def CHF_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def GBP_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def JPY_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def EUR_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'EURO FX - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def AUD_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	AUD_df = COT_df.loc[COT_df.name == 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def NZD_COT():
	COT_df = pd.read_csv('COT_file/FinFutYY.txt')[['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD','Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All']]
	COT_df.columns = ['name','date','long','short']
	df = COT_df.loc[COT_df.name == 'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE']
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.drop(columns=['name'])
	return df

def Selection(code):
	if code == 'CAD':
		return CAD_COT()
	if code == 'CHF':
		return CHF_COT()
	if code == 'GBP':
		return GBP_COT()
	if code == 'JPY':
		return JPY_COT()
	if code == 'EUR':
		return EUR_COT()
	if code == 'AUD':
		return AUD_COT()
	if code == 'NZD':
		return NZD_COT()