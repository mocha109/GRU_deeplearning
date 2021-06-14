# %%
import sys
import time
import matplotlib.pyplot as plt
# from pandas.core.indexes.base import Index
# import seaborn as sns
# from npTOcp import *  # import numpy as np
import numpy as np
from function import clip_grads, clip_STgrads
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
        # batch_t = np.empty((var_size, batch_size, time_size), dtype='f')
        
        for var in range(var_size):
            for i, offset in enumerate(offsets):
                batch_x[var, i, :] = x[offset : offset + time_size, var]
        return batch_x

    def loadParams(self):
        model = self.model
        model.load_params()

    def single_fit(self, batch_x, single_ts, fix_rate, max_epoch=10, max_grad=None, saveP = False):
        '''
        本函式僅適用單一股票模型訓練，請輸入時確認資料型態
        '''
        var_size, batch_size, time_size = batch_x.shape
        self.ppl_list = []
        model, optimizer = self.model, self.optimizer
        st = model.st

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
            clip_STgrads(params, grads,st)
            optimizer.update(params, grads, st)  # 梯度更新方式

            # 評估困惑度 
            # ppl = np.exp(avg_loss)
            ppl = avg_loss
            epoch_time = time.time() - start_time
            print('| epoch %d |  time %d[s] | perplexity %.2f'
                    % (epoch + 1, epoch_time, ppl))
            self.ppl_list.append(float(ppl))

            if saveP:
                model.save_params()

    
    def multi_fit(self, batch_x, multi_ts, fix_rate, max_epoch, max_grad=None, wt_method='industry', saveP = False):
        '''
        本方法有2種調整權重方式，分別為industry、all_market :
        1.industry : 適用於ts資料及全來自同一產業，此方法透過所有股票共用同一權重來找出此產業中的重要影響因素，並排除個別股票的獨特特徵
        2.all_market : 適用於從所有股票中隨機選取的ts資料集，此方法為輪流以各股票訓練模型，最後將各股票的權重平均後導回模型中
        '''
        var_size, BT, out = multi_ts.shape
        self.ppl_list = []
        model, optimizer = self.model, self.optimizer
        st = model.st

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
                        clip_grads(params, grads, max_grad)
                    clip_STgrads(grads,st, fix_rate)
                    optimizer.update(params, grads, st)  # 梯度更新方式
                
                avg_loss = avg_loss / var_size
                # ppl = np.exp(avg_loss)
                ppl = avg_loss
                elapsed_time = time.time() - start_time
                epoch_time = elapsed_time / (epoch+1)*(varcount+1)
                RestTime = epoch_time*(max_epoch*var_size) - elapsed_time
                RestTime = round(RestTime/3600, 2)
                print('| Epoch %d |  Time %d[s] | RestTime %f[s] hours | Perplexity %.2f'
                        % (epoch + 1, elapsed_time, RestTime, ppl))
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
                    clip_STgrads(params, grads,st, fix_rate)
                    optimizer.update(params, grads, st)  # 梯度更新方式

                    # 評估困惑度
                    
                    # ppl = np.exp(avg_loss)
                    ppl = avg_loss
                    elapsed_time = time.time() - start_time
                    epoch_time = elapsed_time / (epoch+1)*(varcount+1)
                    RestTime = epoch_time*(max_epoch*var_size) - elapsed_time
                    RestTime = round(RestTime/3600, 2)
                    print('| VarCount %d | Epoch %d |  Time %d[s] | RestTime %f[s] hours | Perplexity %.2f'
                            % (varcount + 1, epoch + 1, elapsed_time, RestTime, ppl))
                    self.ppl_list.append(float(ppl))
            
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


    def accuracy(self, xs, ts, ori_data, batch_size):
        '''
        滿分為200%
        '''
        model = self.model
        N, BT, O = ts.shape
        accuracy = []
        accuracy_vr = []
        tsmax = np.arange(BT)
        score = 0
        columns = ori_data.columns

        batch_x = self.get_batch(xs, batch_size)
        hs = model.predict(batch_x)
        hs = softmax(hs)
        hs = np.argmax(hs, axis=1)
        #count = 0

        for n in range(N):
            tsmax = np.argmax(ts[n], axis=1)
            # count = 0

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
                    score -= 1
                
                elif hs[t] < 3 and tsmax[t] > 3:
                    score -= 1
                elif hs[t] < 3 and tsmax[t] == 3:
                    score -= 0.5
                elif hs[t] < 3 and tsmax[t] < 3:
                    score += 0.5
                
                accuracy_vr.append(round(score,2)) # /(t+1)
            accuracy.append(accuracy_vr)
            accuracy_vr = []
            score = 0

       
        acc = np.array(accuracy)
        acc = acc.T
        row = int(acc.shape[1] // 3)
        if acc.shape[1]%3 !=0 & row*3<acc.shape[1]:
            rows = row
        else:
            rows = row+1
        
        if acc.shape[1] <= 39:
            plt.figure(figsize=(12,15))
            for pn in range(acc.shape[1]):
                plt.subplot(rows,3,pn+1)
                plt.plot(range(BT), acc[:,pn])
                plt.ylim([np.min(acc), np.max(acc)]) #-1,2
                plt.title(columns[pn])
            plt.tight_layout()

            plt.show()
        elif acc.shape[1] > 39 & acc.shape[1] < 80:
            stock = 39
            rows = 39 // 3
            plt.figure(figsize=(12,15))
            for pn in range(stock):
                plt.subplot(rows,3,pn+1)
                plt.plot(range(BT), acc[:,pn])
                plt.ylim([np.min(acc), np.max(acc)]) #np.min(acc), np.max(acc)
                plt.title(columns[pn])
            plt.tight_layout()

            plt.show()

            stock = acc.shape[1] - 39
            rows = stock // 3
            if stock%3 !=0 & rows*3>stock:
                rows = rows+1
            elif rows==0:
                rows = 1

            plt.figure(figsize=(12,15))
            for pn in range(stock):
                plt.subplot(rows,3,pn+1)
                plt.plot(range(BT), acc[:,pn])
                plt.ylim([np.min(acc), np.max(acc)]) #np.min(acc), np.max(acc)
                plt.title(columns[pn])
            plt.tight_layout()

            plt.show()
        else:
            sys.exit('過多股票，請減少至80檔以下')
        
        return acc


    def summary(self, xs_name, st):
        # , plotting
        model = self.model
        params = model.params
        N = int(params[1][1].shape[0] / 2)

        # 參數數量計算
        number = 0
        for layer in params:
            for w in layer:
                p1 = w.size
                number = number + p1
        
        # affine層整理
        affine_c = params[1][2]

        affinew = np.hstack([np.mean(params[1][0][:,4:N], axis=1), params[1][0][:,3], np.mean(params[1][0][:,:3], axis=1),
                             np.mean(params[1][0][:,(N+5):], axis=1), params[1][0][:,(N+4)], np.mean(params[1][0][:,N:(N+4)], axis=1)])
        affinew = affinew.reshape((-1,6))

        affineb = np.vstack([np.mean(params[1][1][4:N]), params[1][1][3], np.mean(params[1][1][:3]),
                             np.mean(params[1][1][(N+5):]), params[1][1][(N+4)], np.mean(params[1][1][N:(N+4)])])

        affinew = pd.DataFrame(affinew, columns=xs_name, index=['wn_up', 'wn_mid', 'wn_dow', 'wst_up', 'wst_mid', 'wst_dow'])
        affineb = pd.DataFrame(affineb.T, columns=['bn_up', 'bn_mid', 'bn_dow', 'bst_up', 'bst_mod', 'bst_down'])

        stc = np.vstack([affine_c, st.T])
        # stc = stc.T
        stc = pd.DataFrame(stc.T, columns=['threshold','st'])
        # stc = pd.concat([ori_data,stc],axis=1)

        return number, affinew, affineb, stc
            
 #製作中
    def profit(self, xs, labels, ori_data, batch_size, each_buy = 1000,stops= 0.3, good_time = 2, hesitate = 0.5, bed_time= 0.8):
        model = self.model
        N, BT, O = labels.shape
        buyhold = []
        buytime = []
        selltime = []
        profits = []
        all_profits = []
        col_name = list(ori_data.columns)

        batch_x = self.get_batch(xs, batch_size)
        hs = model.predict(batch_x)
        hs = softmax(hs)
        hs = np.argmax(hs, axis=1)

        for n in range(N):
            #tsmax = np.argmax(labels[n], axis=1)
            buyhold.append([col_name[n],0,0])
            buytime.append([])
            selltime.append([])
            profits.append([])

            for t in range(BT):
                price = ori_data.iloc[t,n]

                # 指損判斷
                if (buyhold[n][2] > 0) & (price <= buyhold[n][1]*(1-stops)):
                    ret = round((price - buyhold[n][1]) / buyhold[n][1], 2)
                    earn = round(price*buyhold[n][2] - buyhold[n][1]*buyhold[n][2],2)
                    profits[n].append([earn,ret])
                    buyhold[n][1] = 0
                    buyhold[n][2] = 0
                    selltime[n].append(t)

                # 6:買兩倍
                elif hs[t] == 6:
                    in_ratio = round(good_time*each_buy,0)
                    buytime[n].append(t)
                    buyhold[n][1] = round((buyhold[n][1]*buyhold[n][2] + price*in_ratio) / (buyhold[n][2] + in_ratio),3)
                    buyhold[n][2] = buyhold[n][2] + in_ratio
                
                # 4、5:買1倍
                elif (hs[t] > 3) & (hs[t] < 6):
                    buytime[n].append(t)
                    # if buyhold[n][2] ==0:
                    #     buyhold[n][2] +=1
                    buyhold[n][1] = round((buyhold[n][1]*buyhold[n][2] + price*each_buy) / (buyhold[n][2] + each_buy),3)
                    buyhold[n][2] = buyhold[n][2] + each_buy
                
                # 3:依hesitate調整，決定要售出多少比例股票
                elif (hs[t] == 3) & (buyhold[n][2] > 0):
                    out_ratio = round(hesitate*buyhold[n][2],0)

                    if (hesitate != 0) & (out_ratio !=0):
                        ret = round((price - buyhold[n][1]) / buyhold[n][1], 2)
                        earn = round(price*out_ratio - buyhold[n][1]*out_ratio,2)
                        profits[n].append([earn,ret])
                        buyhold[n][1] = round(buyhold[n][1]*(buyhold[n][2]-out_ratio),3)
                        buyhold[n][2] = buyhold[n][2]-out_ratio
                        selltime[n].append(t)
                
                # 1、2:依bed_time調整，決定要售出多少比例股票
                elif (hs[t] < 3) & (hs[t] > 0) & (buyhold[n][2] > 0):
                    out_ratio = round(bed_time*buyhold[n][2],0)

                    if (bed_time != 0) & (out_ratio !=0):
                        ret = round((price - buyhold[n][1]) / buyhold[n][1], 2)
                        earn = round(price*out_ratio - buyhold[n][1]*out_ratio,2)
                        profits[n].append([earn,ret])
                        buyhold[n][1] = round(buyhold[n][1]*(buyhold[n][2]-out_ratio),3)
                        buyhold[n][2] = buyhold[n][2]-out_ratio
                        selltime[n].append(t)
                
                # 0:全部賣出
                elif (hs[t] == 0) & (buyhold[n][2] != 0):
                    ret = round((price - buyhold[n][1]) / buyhold[n][1], 2)
                    earn = round(price*buyhold[n][2] - buyhold[n][1]*buyhold[n][2],2)
                    profits[n].append([earn,ret])
                    buyhold[n][1] = 0
                    buyhold[n][2] = 0
                    selltime[n].append(t)
        
        pp = 0
        avg = 0
        for n in range(N):
            count = 0
            for i in profits[n]:
                count += 1
                pp = pp + i[0]
                avg = (avg + i[1]) / count
            
            all_profits.append([col_name[n], pp, avg])
                
        return all_profits, profits, buyhold, buytime, selltime



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







