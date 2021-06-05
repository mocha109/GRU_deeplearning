# coding: utf-8
# 資料標籤
import requests
import pandas as pd
from io import StringIO
import datetime
import time
import random
# os : 專門負責文件或目錄處理的軟件
import os  


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

    return listed
listed= DownloadStockID()
listed.head()


#從yfinance抽取30筆
def StockSample(idinfo):
    stock_num =idinfo['有價證券代號']
    stocks=(random.sample(list(stock_num),30))
    df=yf.download(stocks,start="2000-01-01",end="2020-12-31")
    data=df["Adj Close"][stocks]
    print(data)
    print(data.columns)
    
    return df
data=StockSample(idinfo=listed)

#刪除空值
data=data.dropna(axis=0)


#計算調整股價平均
dataa=data.to_numpy()
#test1=data.resample('M').sum()
test2=data.resample('M').mean()
#print(test1)
print(test2)




# 從YAHOO FINANC抓取資料
def stock_load(stock_id, time_start=0, time_end=str(create_today_timestamp()), frequency='d'):
    '''
    1.frquency :　d(日)、wk(週)、mo(月)
    2.time_start=0 在 yahoo finance中表示1970-1-1
    '''
    url = "https://query1.finance.yahoo.com/v7/finance/download/" 
    url = url + stock_id + "?period1=" + str(time_start) + "&period2=" + str(time_end) + "&interval=1" + frequency +"&events=history&includeAdjustedClose=true"

    #抓取資料
    time.sleep(random.randint(0,2))
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text),index_col = "Date",parse_dates = ["Date"])

    return df


# 將資料進行載入與儲存(只處理單一變數)
def stock_data(stock_id, df1, DLaddress="D:\\StockData\\") :
    '''
    請在此輸入各位的股價儲存資料夾位置，以避免每次有人更新都要重找位置
      mocha : D:\\StockData\\
    '''
    #舊資料讀取與新資料儲存
    address = DLaddress + stock_id + ".csv"
    if  os.path.isfile(address):
        df_new = pd.read_csv(address,index_col = "Date",parse_dates = ["Date"])

        if str(create_today_timestamp.today) not in df_new.index:
            # df_new = df_new.append(df)
            df1.to_csv(address,encoding='utf-8')
            print(stock_id + "_已更新到最新資料")

        else:
            print(stock_id + "_已是最新資料，無需更新")
        
    else:
        df1.to_csv(address,encoding='utf-8')
        print(stock_id + "_此為新資料，已創建csv檔")


# 將變數進行迴圈以處理大量股價
def multi_stock():
    stock_num = DownloadStockID()
    count_success = 1
    
    for code in stock_num['有價證券代號'] :   
        try:
            stock_data(stock_id = code, df1 = stock_load(code))
            print("已完成 :" + count_success +"/948")
            count_success += 1
        except:
            print(code + "_fail")

multi_stock()


#list(stock_num)
#stocks = random.sample(list(stock_num),30)
#print(stocks)
#import yfinance as yf
#df = yf.download(stocks,start="2020-07-31",end="2020-12-31")
#print(df.columns)
#print(df['Adj Close'].head())



# stock_num = DownloadStockID()
# count_success = 0

# for code in stock_num['有價證券代號'] :   
#     try:
#         df = stock_load(code)
#         stock_data(stock_id = code, df1 = df)
#         #　print(code + "successful")
#     except:
#         print(code + "_fail")

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


