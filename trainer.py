# import numpy
import time
import matplotlib.pyplot as plt
from npTOcp import *  # import numpy as np
from function import clip_grads

class RnnGRUTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None  # what is this
        self.current_epoch = 0
    
    def get_batch(self, x, batch_size):
        data_size, var_size = x.shape
        time_size = data_size // batch_size
        offsets = [i * time_size for i in range(batch_size)]  # 計算各變數的各批次開始載入的位置

        batch_x = np.empty((var_size, batch_size, time_size), dtype='i')
        batch_t = np.empty((var_size, batch_size, time_size), dtype='i')
        
        for var in range(var_size):
            for i, offset in enumerate(offsets):
                batch_x[var, i, :] = x[var, offset : offset + time_size]
        return batch_x

    def fit(self, xs, single_ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer

        start_time = time.time()
        for epoch in range(max_epoch):
            
            # 將資料形式整理為批次
            batch_x = self.get_batch(xs, batch_size)

            # 計算梯度，更新參數
            avg_loss = model.forward(batch_x, single_ts)# BATCH_X,T不存在，這邊有點難改
            model.backward()
            params, grads = model.params, model.grads
            if max_grad is not None:  # 梯度裁減
                clip_grads(grads, max_grad)
            optimizer.update(params, grads)  # 梯度更新方式

            # 評估困惑度 
            ppl = np.exp(avg_loss)
            elapsed_time = time.time() - start_time
            print('| epoch %d |  time %d[s] | perplexity %.2f'
                    % (self.current_epoch + 1, elapsed_time, ppl))
            self.ppl_list.append(float(ppl))

        self.current_epoch += 1

    def splot(self, max_epoch, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('epoch (x' + str(max_epoch) + ')')
        plt.ylabel('perplexity')
        plt.show()
    
    def multi_fit(self, xs, multi_ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer

        start_time = time.time()
        for epoch in range(max_epoch):
            
            # 將資料形式整理為批次
            batch_x = self.get_batch(xs, batch_size)

            # 計算梯度，更新參數
            avg_loss = model.forward(batch_x, multi_ts)# BATCH_X,T不存在，這邊有點難改
            model.backward()
            params, grads = model.params, model.grads
            if max_grad is not None:  # 梯度裁減
                clip_grads(grads, max_grad)
            optimizer.update(params, grads)  # 梯度更新方式

            # 評估困惑度 
            ppl = np.exp(avg_loss)
            elapsed_time = time.time() - start_time
            print('| epoch %d |  time %d[s] | perplexity %.2f'
                    % (self.current_epoch + 1, elapsed_time, ppl))
            self.ppl_list.append(float(ppl))

        self.current_epoch += 1

# 這邊應該也不用大改
def remove_duplicate(params, grads):
    '''
    把參數陣列中重複的權重整合成一個，並加上對應該權重的梯度
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1): # L藂哪裡來
            for j in range(i + 1, L):
                # 共用權重時
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 當作轉置矩陣，共用權重時（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
