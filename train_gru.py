# %%
# import sys
# sys.path.append('..')
from optimizer import *
from trainer import RnnGRUTrainer
from StockDataDownload import *
from model import *
from StockDataDownload import *
#from npTOcp import *
import numpy as np
import pandas as pd


# %%
# -----------------------------------------------------------------
# 載入總經資料、標籤資料
labels, ori_data = stocklabel(st_amount=60, industry='all', initial_time = "2001-01-01")
ori_data
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 資料載入與整理
xs, st = xs_data(file='C:\\Users\\z1244\\Desktop\\data')
xs = xs_rolling(xs,roll=5,sh=1,axis=0)
xs, labels, ori_data, st = alter_a(xs,labels,ori_data, st)
xs_name = xs.columns

xs = PdNp(xs)
st = PdNp(st)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 訓練與驗證資料區分
xs, labels, ori_data, st, xs_v, labels_v, ori_data_v, st_v = TestValidate(xs, labels, ori_data, st, test_size = 178, batch_size = 10)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
#設定超參數
batch_size = 10
data_size, var_size = xs.shape
output_size = 7
time_size = data_size // batch_size
hidden_size = time_size  # 本模型必要設置，不這樣設在AFFINE層會出錯
lr = 0.1
max_epoch = 30
# max_grad = 
gamma = np.std(xs, ddof=1, axis=0, dtype='f')
st_gamma = np.std(st, ddof=1,dtype='f')
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 產生模型
model = Rnngru(st, gamma, st_gamma, var_size, batch_size, time_size, output_size, hidden_size)
optimizer = Adam(lr)
trainer = RnnGRUTrainer(model, optimizer)

# 套用梯度裁減並學習
batch_x = trainer.get_batch(xs, batch_size)
# trainer.single_fit(batch_x, single_ts, max_epoch=10, batch_size=20, max_grad=None)
trainer.multi_fit(batch_x, max_epoch=max_epoch, multi_ts=labels, max_grad=None, wt_method='all_market') #industry
trainer.plot(max_epoch, ylim=(0, 500))
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 準確度計算
accuracy = trainer.accuracy(xs, labels, ori_data, batch_size=batch_size)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 模型摘要
number, affinew, affineb, stc = trainer.summary(xs_name, st)
stc.plot()
# -----------------------------------------------------------------
# %%
affineb
# %%
# -----------------------------------------------------------------
# 報酬率
all_profits, profits, buyhold, buytime, selltime = trainer.profit(xs, labels, 
                     ori_data, batch_size = batch_size,
                     each_buy = 1000,stops= 0.2, good_time = 3, 
                     hesitate = 1, bed_time= 1)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 用測試資料評估
model.reset_state()
ppl_test = trainer.ppl_list
# print('test perplexity: ', ppl_test)

# 儲存參數
# model.save_params()
# -----------------------------------------------------------------
# %%
a=0
for i in all_profits:
    a = a + i[1]
a

#buyhold
#profits
#buytime

# %%
# ---------------------------------
# TimeSoftmaxWithLoss.
#trainer.model.TimeSoftmaxWithLoss.test
#model.params[0][2]
# ---------------------------------
#np.mean(st)
# c = pd.DataFrame(c,columns=xs_name, index=['ab','cd'])
# c
#model.params[1][0]
#st.shape
#model.layers[0].h
#model.layers[2].cache
#model.loss_layer.cache[1].shape
# %%
