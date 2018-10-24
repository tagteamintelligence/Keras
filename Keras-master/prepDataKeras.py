import csv
import pandas as pd
from functools import reduce

df_usd = pd.read_csv("Data/EUR_USD_H4.csv")
df_aud = pd.read_csv("Data/EUR_AUD_H4.csv")
df_cad = pd.read_csv("Data/EUR_CAD_H4.csv")
df_chf = pd.read_csv("Data/EUR_CHF_H4.csv")
df_gbp = pd.read_csv("Data/EUR_GBP_H4.csv")
df_nzd = pd.read_csv("Data/EUR_NZD_H4.csv")

df_list = [df_usd, df_aud, df_cad, df_chf, df_gbp, df_nzd]
edited_df = []

for x in df_list:
	del x['result']
	edited_df.append(x)

df_usd, df_aud, df_cad, df_chf, df_gbp, df_nzd = edited_df

df_final = pd.concat([df_usd, df_aud, df_cad, df_chf, df_gbp, df_nzd], axis=1)

df_final.to_csv("Data/EUR_Keras.csv", sep=',', encoding='utf-8', index=False)