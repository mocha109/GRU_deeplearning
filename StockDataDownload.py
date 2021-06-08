# coding: utf-8
# 資料標籤
import requests
import pandas as pd
import numpy as np
from io import StringIO
import datetime
import time
import random
import os    # os : 專門負責文件或目錄處理的軟件
import yfinance as yf


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
def DownloadStockID(industry='all', itype='股票', initial_time='2000-01-01'):
    '''
    initial_time為上市時間至少在指定時段就存在的公司
    '''
    url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=&industry_code=&Page=1&chklike=Y"
    response = requests.get(url)
    listed = pd.read_html(response.text)[0]
    listed.columns = listed.iloc[0,:]
    listed = listed[["有價證券代號", "有價證券名稱", "市場別", "有價證券別", "產業別", "公開發行/上市(櫃)/發行日"]]
    listed = listed.iloc[1:]
    listed["有價證券代號"] = listed["有價證券代號"].apply(lambda x: str(x) + ".TW")
    listed = listed[listed["有價證券別"] == itype]
    
    if not industry == 'all':
        listed = listed[listed["產業別"] == industry]

    # 轉換發行日格式
    listed["公開發行/上市(櫃)/發行日"] = pd.to_datetime(listed["公開發行/上市(櫃)/發行日"])
    
    # 自動輸入時間調整資料範圍
    # initial_time = input("輸入日期 : ")
    listed = listed[listed["公開發行/上市(櫃)/發行日"] < datetime.datetime.strptime( str(initial_time) , '%Y-%m-%d' )]
    return listed



#從yfinancen抽取30筆
def StockSample(idinfo, interval = "1d", st_amount=30, start='1990-01-01'):
    '''
    interval(有效期間_：1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 
    '''
    stock_num =idinfo['有價證券代號']
    stocks=(random.sample(list(stock_num),st_amount))
    df=yf.download(stocks,interval=interval,start=start)
    data=df["Adj Close"]
  
    return data


#標籤
def stocklabel(st_amount=30, industry='all', itype='股票', interval= "1d", initial_time= '2000-01-01'):
    '''
    1.interval(有效期間_：1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    2.initial_time為上市時間至少在指定時段就存在的公司，並非返回資料的初始時間
    '''
    listed = DownloadStockID(industry, itype, initial_time)
    data=StockSample(listed, interval, st_amount)
    
    #刪除空值
    data=data.dropna(axis=0)
    
    #周期數移動索引
    data=round((data-data.shift(1))/data.shift(1),4)
    data=data.dropna(axis=0)
    
    #將日資料平均成月資料
    data = data.resample('MS').mean()

    ori_data = data.copy()
    data['code']=0
    
    c= len(data.columns)-1
    t = len(data)
    clist = list(data.columns)[:-1]

    labels = np.arange(c*t*7).reshape(c,t,7)
    labels = labels*0
    
    count_id = 0

    for id in clist:
        data['code'][data[id] < -0.1 ] = 0
        data['code'][(data[id] >= -0.1) & (data[id] <-0.05)] = 1
        data['code'][(data[id] >= -0.05) & (data[id] <-0.01)] = 2
        data['code'][(data[id] >= -0.01) & (data[id] <0.01)] = 3
        data['code'][(data[id] >= 0.01) & (data[id] <0.05)] = 4
        data['code'][(data[id] >= 0.05) & (data[id] <0.1)] = 5
        data['code'][data[id] > 0.1] = 6
    
        for time in range(t):
            labels[count_id, time, int(data.iloc[time, list(data.columns).index('code')])] = 1
    
        count_id += 1

    return labels, ori_data

a,b = stocklabel()


def alter_a (a=a,b=b):
    xs=pd.read_csv('data.csv')
    xs['Date']= pd.to_datetime(xs['Date']) 
    xs.set_index(['Date'],inplace=True)
    xs=xs.dropna(axis=1)
    
    
    b1=b.index.strftime("%Y-%m-%d")
    xs1=xs.index.strftime("%Y-%m-%d")
     #此暫時留下，但應該不需要
    
    first_time_xs = xs1[0]
    ind = list(b1).index(first_time_xs)
    last_time_xs = xs1[-1]
    ind_last = list(b1).index(last_time_xs)
    
    
    a = a[:, (ind+1):(ind_last+2) , :]
    return a

print(a)


#讀總經資料
def xs_data(file="C:\\Users\\user\\Desktop\\pyhton2\\datasets"):
    address = []
    h= None
    file_name = os.listdir(file)

    for i in range(len(file_name)):
        add = "{}\\{}".format(file,file_name[i])
        address.append(add)

        for j in address:
            xs=pd.read_csv(j,index_col=0)
            

            if h ==None: 
                h= xs
            else:
                h = pd.concat([a,xs], axis=1)

    h = h.dropna(axis=0)

    return h 


def xs_rolling(xs,roll=1,sh=1,axis=0):
    
    original=xs.rolling(window=roll,axis=axis).mean()
    original=round((original-original.shift(sh))/original.shift(sh),4)
    original =original.dropna(axis=0)
    return original

original=xs_rolling(a,roll=3)
original




# 從YAHOO FINANC抓取資料
#def stock_load(stock_id, time_start=0, time_end=str(create_today_timestamp()), frequency='d'):
 #   '''
  #  1.frquency :　d(日)、wk(週)、mo(月)
   # 2.time_start=0 在 yahoo finance中表示1970-1-1
  #  '''
 #   url = "https://query1.finance.yahoo.com/v7/finance/download/" 
 #   url = url + stock_id + "?period1=" + str(time_start) + "&period2=" + str(time_end) + "&interval=1" + frequency +"&events=history&includeAdjustedClose=true"

    #抓取資料
#   time.sleep(random.randint(0,2))
#  response = requests.get(url)
#   df = pd.read_csv(StringIO(response.text),index_col = "Date",parse_dates = ["Date"])

 #   return df


# 將資料進行載入與儲存(只處理單一變數)
#def stock_data(stock_id, df1, DLaddress="D:\\StockData\\") :
#    '''
#    請在此輸入各位的股價儲存資料夾位置，以避免每次有人更新都要重找位置
#      mocha : D:\\StockData\\
#    '''
    #舊資料讀取與新資料儲存
 #   address = DLaddress + stock_id + ".csv"
 #   if  os.path.isfile(address):
 #       df_new = pd.read_csv(address,index_col = "Date",parse_dates = ["Date"])

 #       if str(create_today_timestamp.today) not in df_new.index:
            # df_new = df_new.append(df)
 #           df1.to_csv(address,encoding='utf-8')
 #           print(stock_id + "_已更新到最新資料")

 #       else:
 #           print(stock_id + "_已是最新資料，無需更新")
        
#   else:
#        df1.to_csv(address,encoding='utf-8')
#        print(stock_id + "_此為新資料，已創建csv檔")


# 將變數進行迴圈以處理大量股價
#def multi_stock():
#    stock_num = DownloadStockID()
#    count_success = 1
    
#    for code in stock_num['有價證券代號'] :   
#        try:
#            stock_data(stock_id = code, df1 = stock_load(code))
#            print("已完成 :" + count_success +"/948")
#            count_success += 1
#        except:
#            print(code + "_fail")

#multi_stock()