# coding: utf-8
# 資料標籤
import sys
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
def stocklabel(st_amount=30, return_data = 1, industry='all', itype='股票', interval= "1d", initial_time= '2000-01-01', start='1990-01-01'):
    '''
    1.interval       :(有效期間_：1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max (有問題，暫時別調整此參數)，default='1d'
    2.initial_time   :為上市時間至少在指定時段就存在的公司，並非返回資料的初始時間，default='2000-01-01'
    3.start          :為yfinance下載資料的起始時段，default='1990-01-01'
    4.st_amount      :用於調整一次抓取的股票數量，default=30
    5.return_data    :用於調整您要計算的股票報酬率為幾日報酬率，default=1
    6.itype          :選擇您要下載的資料為股票或其他金融資料，default='股票'
    7.industry       :選擇您希望下載的產業，default='all'
    '''
    listed = DownloadStockID(industry, itype, initial_time)
    data=StockSample(listed, interval, st_amount, start)
    
    #刪除空值
    data=data.dropna(axis=0)
    
    # #計算報酬率
    # data=round((data-data.shift(return_data))/data.shift(return_data),4)
    # data=data.dropna(axis=0)
    
    #將日資料平均成月資料
    data = data.resample('MS').mean()
    ori_data = data.copy()
    ori_data = ori_data.dropna(axis=0)

    #計算報酬率
    data=round((data-data.shift(return_data))/data.shift(return_data),4)
    data=data.dropna(axis=0)

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



def alter_a(xs, labels, ori_data, st):
    '''
    1.將xs與標籤資料對期
    2.labels,ori_data來自stocklabel函數
    '''
    
    # 統一xs與ori_data的index格式以方便對比
    ori_data.index = pd.to_datetime(ori_data.index, format="%Y-%m-%d")
    xs.index = pd.to_datetime(xs.index, format="%Y-%m-%d")
    st.index = pd.to_datetime(st.index, format="%Y-%m-%d")

    b1 = list(ori_data.index.strftime("%Y-%m-%d"))
    xs1 = list(xs.index.strftime("%Y-%m-%d"))
    st1 = list(st.index.strftime("%Y-%m-%d"))

    # 用concat尋找共同時間點，最後輸出一致時間長度的xs,labels,st
    all_df = pd.concat([ori_data,xs,st],axis=1)
    all_df_remove_na = all_df.dropna(axis=0)
    all_df_remove_na1=list(all_df_remove_na.index.strftime("%Y-%m-%d"))

    first_time_all = all_df_remove_na1[0]
    last_time_all = all_df_remove_na1[-1]
    
    ind = b1.index(first_time_all)
    ind_last = b1.index(last_time_all)
    labels = labels[:, (ind+1):(ind_last+2) , :]

    ind = b1.index(first_time_all)
    ind_last = b1.index(last_time_all)
    ori_data = ori_data[(ind+1):(ind_last+2)]
    
    ind = st1.index(first_time_all)
    ind_last = st1.index(last_time_all)
    st = st[ind:ind_last]
    
    ind = xs1.index(first_time_all)
    ind_last = xs1.index(last_time_all)
    xs = xs[ind:ind_last]

    xs = xs.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    st = (st - np.mean(st)) / np.std(st)

    return xs, labels, ori_data, st



#讀總經資料
def xs_data(file="C:\\Users\\user\\Desktop\\pyhton2\\datasets"):
    '''
    file為您存放總體經濟資料的資料夾位置，請注意資料夾內只能有csv檔
    '''
    address = []
    first= None
    count = 0
    file_name = os.listdir(file)

    for i in range(len(file_name)):
        add = "{}\\{}".format(file,file_name[i])
        address.append(add)

    for j in address:
        if j != "{}\\st.csv".format(file):
            xs = pd.read_csv(j,index_col=0, header=0, parse_dates=['Date'],thousands=",")
            
            if count == 0: 
                first= xs
                count = count + 1
            else:
                first = pd.concat([first,xs], axis=1)
        else:
            st = pd.read_csv(j,index_col=0, usecols=[0, 1],
                                header=0,parse_dates=['Date'])

    first = first.dropna(axis=0)

    return first, st


def xs_rolling(xs,roll=1,sh=1,axis=0):
    '''
    1.xs資料格式為dataframe
    2.將資料進行平滑並計算報酬率
    '''
    
    original=xs.rolling(window=roll,axis=axis).mean()
    original=round((original-original.shift(sh))/original.shift(sh),4)
    original =original.dropna(axis=0)

    return original


def PdNp(xs, pdTOnp = True):
    xs = xs.to_numpy()

    return xs


def TestValidate(xs, labels, ori_date, st, test_size, batch_size):
    print('目前資料總長度為:\ntest : {}，validate : {}\nNOTE.1:設定時請注意validate長度一定要可被batch_size整除\n\nNOTE.2:若您的validate長度不合規範，可在input中直接輸入新的test_size'.format(len(xs[:test_size]), len(xs[test_size:])))
    confirm = input("請確認您的test_size，確定是否要繼續執行訓練資料集切割(Y / New test_size): ")
    
    if confirm != 'Y':
        test_size = int(confirm)
        confirm = 'Y'

    if confirm == 'Y':
        if len(xs[test_size:]) % batch_size == 0:
            minus = (len(xs[:test_size]) % batch_size)

            xs_test = xs[minus:test_size]
            xs_validate = xs[test_size:]
            labels_test = labels[:, minus:test_size, :]
            labels_validate = labels[:, test_size: , :]
            ori_date_test = ori_date[minus:test_size]
            ori_date_validate = ori_date[test_size:]
            st_test = st[minus:test_size]
            st_validate = st[test_size:]
            
            print('目前資料總長度為test : {}，validate : {}'.format(len(xs_test), len(xs_validate)))

            return xs_test, labels_test, ori_date_test, st_test, xs_validate, labels_validate, ori_date_validate, st_validate
        
        else:
            sys.exit("中斷執行，請確認驗證集資料筆數可被batch_size整除")
    
    else:
        sys.exit("中斷執行，請確認訓練集大小後重新執行本函數")

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