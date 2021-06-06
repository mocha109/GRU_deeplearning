import sys
sys.path.append('..')
from optimizer import SGD
from trainer import GRUTrainer
from DataFrame import 
from model import Rnngru


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


# 產生模型
model = Rnngru(var_size, batch_size, time_size, output_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnGRUTrainer(model, optimizer)

# 套用梯度裁減並學習
trainer.fit(xs, single_ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20)
trainer.multi_fit(xs, multi_ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
trainer.plot(ylim=(0, 500))

# 用測試資料評估
model.reset_state()
ppl_test = eval_perplexity(model, )
print('test perplexity: ', ppl_test)

# 儲存參數
model.save_params()
