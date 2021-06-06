import requests
import pandas as pd
import numpy as np
from io import StringIO
import datetime
import time
import random
# os : 專門負責文件或目錄處理的軟件
import os 
import yfinance as yf
from matplotlib import pyplot as plt
from collections import Counter


# 時間處理函數(yahoo finance時間為數值格式，因此不能直接輸入'2021-5-31'這類格式)
def create_today_timestamp():
    '''
    1.time.gmtime : https://www.runoob.com/python/att-time-gmtime.html
    2.time.strftime : https://www.runoob.com/python/att-time-strftime.html
    3.time.mktime : https://www.runoob.com/python/att-time-mktime.html
    4.time.strptime : https://www.runoob.com/python/att-time-strptime.html
    '''
    today = time.strftime("%Y-%m-%d",time.gmtime())
    return int(time.mktime(time.strptime(today, "%Y-%m-%d")))

def create_timestamp_from_today(n):
    '''
    此函數可以返回今天再加n，暫時用不到
    '''
    today = create_today_timestamp()
    return today + n*24*3600


    # 從"台灣證券交易所"下載上市公司代號(包含期貨、etf)
def DownloadStockID(industry='all', type='股票'):
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0,:]
    listed = listed[["有價證券代號", "有價證券名稱", "市場別", "有價證券別", "產業別", "公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]
    listed["有價證券代號"] = listed["有價證券代號"].apply(lambda x: str(x) + ".TW")
    listed = listed[listed["有價證券別"] == type]
    
    if not industry == 'all':
        listed = listed[listed["產業別"] == industry]
    listed["公開發行/上市(櫃)/發行日"] = pd.to_datetime(listed["公開發行/上市(櫃)/發行日"])
    time_start = input("輸入日期 : ")
    listed = listed[listed["公開發行/上市(櫃)/發行日"] < datetime.datetime.strptime( str(time_start) , '%Y-%m-%d' )]
    return listed

listed= DownloadStockID()
listed

        #df[stockdate > '2005-01-01']
        #listed =stockdate.truncate(before='2005-01-01')
              #listed = listed[listed["公開發行/上市(櫃)/發行日"] <= basetime]
        #basetime=stockdata.truncate(before='2005-01-01')
        #print(df.truncate(before='2005-01-01'))


#從yfinance抽取30筆
def StockSample(idinfo):
    stock_num =idinfo['有價證券代號']
    stocks=(random.sample(list(stock_num),30))
    df=yf.download(stocks,interval = "1mo" )# 有效期間：1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max  
    data=df["Adj Close"][stocks]
    print(data)
    print(data.columns)
    
    return data

data=StockSample(idinfo=listed)


#刪除空值
data=data.dropna(axis=0)
data

#周期數移動索引
data=(data-data.shift(1))/data
data=data.dropna(axis=0)
data

b = np.arange(len(data)*len(data.columns)).reshape(len(data),len(data.columns))
a = pd.DataFrame(b,index=data.index)
a
a['1'][data['2012.TW'] < 0.1] = 1