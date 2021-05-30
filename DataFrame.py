# coding: utf-8
# 資料標籤
import requests
import pandas as pd
from io import StringIO
import datetime
# os : 專門負責文件或目錄處理的軟件
import os  

#從"台灣證券交易所"下載上市公司代號(包含期貨、etf)
def DownloadStockID():
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0,:]
    listed = listed[["有價證券代號","有價證券名稱","市場別","產業別","公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]
    # listed
    stock_1 = listed["有價證券代號"]
    stock_num = stock_1.apply(lambda x: str(x) + ".TW")

    return stock_num


# 抓資料，並讀寫成csv檔
def stock_data(stock_id, time_start, time_end, DLaddress="C:\\Users\\user\\Desktop\\pyhton2\\stock\\") :
    days = 24 * 60 * 60    #一天有86400秒 
    initial = datetime.datetime.strptime( '1970-01-01' , '%Y-%m-%d' )
    start = datetime.datetime.strptime( time_start , '%Y-%m-%d' )
    end = datetime.datetime.strptime( time_end, '%Y-%m-%d' )
    period1 = start - initial
    period2 = end - initial
    s1 = period1.days * days
    s2 = period2.days * days

    url = "https:\\query1.finance.yahoo.com\\v7\\finance\\download\\" 
    url = url + stock_id + "?period1=" + str(s1) + "&amp;period2=" + str(s2) + "&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true"

    #抓取資料
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text),index_col = "Date",parse_dates = ["Date"])

    #舊資料讀取與新資料儲存
    address = DLaddress + stock_id + ".csv"
    if  os.path.isfile(address):
        df_new = pd.read_csv(address,index_col = "Date",parse_dates = ["Date"])

        if time_start not in df_new.index:
            df_new = df_new.append(df)
            df_new.to_csv(address,encoding='utf-8')
            print("已更新到最新資料")

        else:
            print("已是最新資料，無需更新")
        
    else:
        df.to_csv(address,encoding='utf-8')
        print("此為新資料，已創建csv檔")



#資料執行
time_start = "2000-01-01"
time_end = "2020-12-31"

stock_num = DownloadStockID()

for i in stock_num :   
    try:
        stock_data(i, time_start,time_end)
        print(i + "successful")
    except:
        print(i + "fail")

# 自動處理資料(空值資料、貼標籤)





# 股價處理
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# Exponential moving average
    # Params: 
        # data: pandas DataFrame
        # period: smoothing period
        # column: the name of the column with values for calculating EMA in the 'data' DataFrame

# def ema(data, period=0, column='<CLOSE>'):
    # data['ema' + str(period)] = data[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()
    
    # return data


