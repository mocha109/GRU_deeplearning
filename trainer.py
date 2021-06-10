# %%
# import numpy
import time
import matplotlib.pyplot as plt
import seaborn as sns
# from npTOcp import *  # import numpy as np
import numpy as np
from function import clip_grads
from model import *
import pandas as pd

class RnnGRUTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        # self.current_epoch = 0
        # self.current_var = 0
    
    def get_batch(self, x, batch_size):
        '''
        用於將輸入的總經變數從BT*N格式轉為N*B*T格式
        '''
        data_size, var_size = x.shape
        time_size = data_size // batch_size
        offsets = [i * time_size for i in range(batch_size)]  # 計算各變數的各批次開始載入的位置

        batch_x = np.empty((var_size, batch_size, time_size), dtype='f')
        batch_t = np.empty((var_size, batch_size, time_size), dtype='f')
        
        for var in range(var_size):
            for i, offset in enumerate(offsets):
                batch_x[var, i, :] = x[offset : offset + time_size, var]
        return batch_x

    def loadParams(self):
        model = self.model
        model.load_params()

    def single_fit(self, batch_x, single_ts, max_epoch=10, max_grad=None, saveP = False):
        '''
        本函式僅適用單一股票模型訓練，請輸入時確認資料型態
        '''
        var_size, batch_size, time_size = batch_x.shape
        self.ppl_list = []
        model, optimizer = self.model, self.optimizer

        start_time = time.time()
        for epoch in range(max_epoch):
            
            # 將資料形式整理為批次
            # batch_x = self.get_batch(xs, batch_size)

            # 計算梯度，更新參數
            avg_loss = model.forward(batch_x, single_ts)
            model.backward()
            params, grads = model.params, model.grads
            if max_grad is not None:  # 梯度裁減
                clip_grads(grads, max_grad)
            optimizer.update(params, grads)  # 梯度更新方式

            # 評估困惑度 
            ppl = np.exp(avg_loss)
            epoch_time = time.time() - start_time
            print('| epoch %d |  time %d[s] | perplexity %.2f'
                    % (epoch + 1, epoch_time, ppl))
            self.ppl_list.append(float(ppl))

            if saveP:
                model.save_params()

    
    def multi_fit(self, batch_x, multi_ts, max_epoch=10, max_grad=None, wt_method='industry', saveP = False):
        '''
        本方法有2種調整權重方式，分別為industry、all_market :
        1.industry : 適用於ts資料及全來自同一產業，此方法透過所有股票共用同一權重來找出此產業中的重要影響因素，並排除個別股票的獨特特徵
        2.all_market : 適用於從所有股票中隨機選取的ts資料集，此方法為輪流以各股票訓練模型，最後將各股票的權重平均後導回模型中
        '''
        var_size, batch_size, time_size = batch_x.shape
        self.ppl_list = []
        model, optimizer = self.model, self.optimizer
        all_params = None
        all_grads = None

        # 將資料形式整理為批次
        # batch_x = self.get_batch(xs, batch_size)

        start_time = time.time()
        if wt_method == 'industry':
            for epoch in range(max_epoch):

                for varcount in range(var_size):
                    single_ts = multi_ts[varcount]

                    # 計算梯度，更新參數
                    avg_loss = model.forward(batch_x, single_ts)
                    model.backward()
                    params, grads = model.params, model.grads
                    if max_grad is not None:  # 梯度裁減
                        clip_grads(grads, max_grad)
                    optimizer.update(params, grads)  # 梯度更新方式

                    # 評估困惑度 
                    ppl = np.exp(avg_loss)
                    elapsed_time = time.time() - start_time
                    epoch_time = elapsed_time / (epoch+1)*(varcount+1)
                    RestTime = epoch_time*(max_epoch*var_size) - elapsed_time
                    RestTime = round(RestTime/3600, 2)
                    print('| VarCount %d | Epoch %d |  Time %d[s] | RestTime %f[s] hours | Perplexity %.2f'
                            % (varcount + 1, epoch + 1, elapsed_time, RestTime, ppl))
                    self.ppl_list.append(float(ppl))
        
            if saveP:
                    model.save_params()
        
        elif wt_method == 'all_market':
            for varcount in range(var_size):
                single_ts = multi_ts[varcount]

                for epoch in range(max_epoch):
                    # 計算梯度，更新參數
                    avg_loss = model.forward(batch_x, single_ts)
                    model.backward()
                    params, grads = model.params, model.grads
                    if max_grad is not None:  # 梯度裁減
                        clip_grads(grads, max_grad)
                    optimizer.update(params, grads)  # 梯度更新方式

                    # 評估困惑度 
                    ppl = np.exp(avg_loss)
                    elapsed_time = time.time() - start_time
                    epoch_time = elapsed_time / (epoch+1)*(varcount+1)
                    RestTime = epoch_time*(max_epoch*var_size) - elapsed_time
                    RestTime = round(RestTime/3600, 2)
                    print('| VarCount %d | Epoch %d |  Time %d[s] | RestTime %f[s] hours | Perplexity %.2f'
                            % (varcount + 1, epoch + 1, elapsed_time, RestTime, ppl))
                    self.ppl_list.append(float(ppl))
                
                # 將訓練完的權重平均
                all_params += params
                all_grads += grads
            
            # 將平均權重放回模型內部
            model.params, model.grads = all_params/var_size, all_grads/var_size
            all_params = None
            all_grads = None
            
            if saveP:
                model.save_params()
            
        else:
            print('請輸入正確wt_method(可用選項 : industry、all_market)')
            

        print('模型訓練結束，總計耗時:{}'.format(time.time() - start_time))


    def plot(self, max_epoch, ylim=None):
        '''
        
        '''
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('epoch (x' + str(max_epoch) + ')')
        plt.ylabel('perplexity')
        plt.show()


    def accuracy(self, batch_size, xs, ts, columns):
        '''
        滿分為200%
        '''
        model = self.model
        N, BT, O = ts.shape
        accuracy = []
        tsmax = np.arange(BT)
        score = 0

        batch_x = self.get_batch(xs, batch_size)
        hs = model.predict(batch_x)
        hs = np.argmax(hs, axis=1)

        for n in range(len(ts)):
            tsmax = np.argmax(ts[n], axis=1)
            for t in range(BT):
                if hs[t] == tsmax[t]:
                    score += 2
                elif hs[t] > 3 and tsmax[t] > 3:
                    score += 0.5
                elif hs[t] > 3 and tsmax[t] == 3:
                    score -= 0.5
                elif hs[t] > 3 and tsmax[t] < 3:
                    score -= 1
                
                elif hs[t] == 3 and tsmax[t] != 3:
                    score -= 0.5
                
                elif hs[t] < 3 and tsmax[t] > 3:
                    score -= 1
                elif hs[t] < 3 and tsmax[t] == 3:
                    score -= 0.5
                elif hs[t] < 3 and tsmax[t] < 3:
                    score += 1
            accuracy.append(round(score/BT,2))

        plt.plot(columns, accuracy, label='Accuracy for each stock')
        plt.xlabel('StockID')
        plt.ylabel('Accuracy')
        plt.show()
        
        return accuracy


    def summary():
        model = self.model
        params = model.params

        affine_nw = params[2,]

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







