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


# %%
# 載入總經資料、標籤資料
labels, ori_data = stocklabel()
xs, st = xs_data(file='C:\\Users\\z1244\\Desktop\\data')
xs = xs_rolling(xs,roll=1,sh=1,axis=0)
xs, labels, ori_data, st = alter_a(xs,labels,ori_data, st)

xs = PdNp(xs)
st = PdNp(st)

xs, labels, ori_data, st, xs_v, labels_v, ori_data_v, st_v = TestValidate(xs, labels, ori_data, st, 120, batch_size = 20)

# %%
#設定超參數
batch_size = 10
data_size, var_size = xs.shape
output_size = 7
hidden_size = 100
time_size = data_size // batch_size
lr = 0.05
max_epoch = 10
# max_grad = 
gamma = np.std(xs, ddof=1)
st_gamma = np.std(st, ddof=1)

# %%
# 產生模型
model = Rnngru(st, gamma, st_gamma, var_size, batch_size, time_size, output_size, hidden_size)
optimizer = Adam(lr)
trainer = RnnGRUTrainer(model, optimizer)

# 套用梯度裁減並學習
# trainer.single_fit(xs, single_ts, max_epoch=10, batch_size=20, max_grad=None)
trainer.multi_fit(xs, multi_ts=labels, max_epoch=10, batch_size=20, max_grad=None, wt_method='industry')
trainer.plot(max_epoch, ylim=(0, 500))

# %%
# 用測試資料評估
model.reset_state()
ppl_test = trainer.ppl_list
print('test perplexity: ', ppl_test)

# 儲存參數
model.save_params()


# %%
