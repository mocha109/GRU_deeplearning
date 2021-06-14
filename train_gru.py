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
import matplotlib.pyplot as plt


# %%
# -----------------------------------------------------------------
# 載入總經資料、標籤資料
labels, ori_data = stocklabel(st_amount=30, industry='all', initial_time = "2001-01-01")
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
xs, labels, ori_data, st, xs_v, labels_v, ori_data_v, st_v = TestValidate(xs, labels, ori_data, st, test_size = 142, batch_size = 10)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
#設定超參數123
batch_size = 10
data_size, var_size = xs.shape
output_size = 7   # 本模型必要設置，不這樣設在AFFINE層會出錯
time_size = data_size // batch_size   # 本模型必要設置，不這樣設在AFFINE層會出錯
hidden_size = time_size   # 本模型必要設置，不這樣設在AFFINE層會出錯

wt_method = 'industry'   # 模型訓練方式(industry、all_market)
lr = 0.01   # 學習率
st_lim = 0.001   # 目前沒用
fix_rate = 0.01   # 當門檻值極值超過st極大與極小值時，要將超過的門檻值縮小的比例
max_epoch = 60   # 整體model訓練次數(注:採用不同訓練方式會有不同)
max_grad = None   # 主參數是否要進行梯度裁減，請輸入數字(none表示不要裁減)

gamma = np.std(xs, ddof=1, axis=0, dtype='f')
st_gamma = np.std(st, ddof=1,dtype='f')
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 產生模型
model = Rnngru(st, gamma, st_gamma, var_size, batch_size, time_size, output_size, hidden_size)
optimizer = Adam(st_lim, lr)
trainer = RnnGRUTrainer(model, optimizer)

# 套用梯度裁減並學習
batch_x = trainer.get_batch(xs, batch_size)
# trainer.single_fit(batch_x, single_ts, max_epoch=10, batch_size=20, max_grad=None)
trainer.multi_fit(batch_x, fix_rate = fix_rate, max_epoch=max_epoch, multi_ts=labels, max_grad=max_grad, wt_method=wt_method)
ppl_test = trainer.ppl_list

if wt_method == 'industry':
    trainer.plot(max_epoch, ylim=((np.min(ppl_test)-1), (np.max(ppl_test)+1)))
else:
    trainer.plot(max_epoch, ylim=(0, (500)))
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 準確度計算
accuracy = trainer.accuracy(xs, labels, ori_data, batch_size=batch_size)
# %%
accuracy = np.array(accuracy)

agg_accuracy = np.zeros(data_size).astype('f')
for row in range(accuracy.shape[1]):
    agg_accuracy = agg_accuracy + accuracy[:,row]
agg_accuracy = agg_accuracy / 60

plt.figure(figsize=(8,6))
plt.plot(agg_accuracy)
# -----------------------------------------------------------------
# %%
# -----------------------------------------------------------------
# 模型摘要
number, affinew, affineb, stc = trainer.summary(xs_name, st)
stc.plot()
# -----------------------------------------------------------------
# %%
affinew
#affineb
#number
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
# a=0
# for i in all_profits:
#     a = a + i[1]
# a

#buyhold
#profits
#buytime
# %%
#buyhold


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
plt.plot(model.layers[2].transition)
#model.loss_layer.cache[1].shape
# %%
