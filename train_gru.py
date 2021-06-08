# import sys
# sys.path.append('..')
from optimizer import *
from trainer import RnnGRUTrainer
from StockDataDownload import *
from model import *
from 

#設定超參數
batch_size = 
var_size = 
output_size = 
hidden_size = 
time_size = 
lr = 
max_epoch = 
max_grad = 

# 載入學習資料
xs = 

# 產生模型
model = Rnngru(st, gamma, st_gamma, var_size, batch_size, time_size, output_size, hidden_size)
optimizer = Adam(lr)
trainer = RnnGRUTrainer(model, optimizer)

# 套用梯度裁減並學習
trainer.single_fit(xs, single_ts, max_epoch=10, batch_size=20, max_grad=None)
trainer.multi_fit(xs, multi_ts, max_epoch=10, batch_size=20, max_grad=None, wt_method='industry')
trainer.plot(max_epoch, ylim=(0, 500))

# 用測試資料評估
model.reset_state()
ppl_test = trainer.ppl_list
print('test perplexity: ', ppl_test)

# 儲存參數
model.save_params()
