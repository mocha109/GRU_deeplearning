#from npTOcp import *
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gru(x,gamma):
    # 此GAMMA為一1*N矩陣
    gamma = 1 / gamma
    return 1 / (1 + np.exp(-gamma * x.T).T)

def sigmoid_st(st,st_gamma,c):
    # 此一GAMMA為純值
    st_gamma = 1 / st_gamma
    return 1 / (1 + np.exp(-st_gamma*(st.T-c).T)) 


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)  # 防止指數溢位
    x = np.exp(x)
    x = x / x.sum(axis=1, keepdims=True)

    # if x.ndim == 2:
    #     x = x - x.max(axis=1, keepdims=True)
    #     x = np.exp(x)
    #     x /= x.sum(axis=1, keepdims=True)
    # elif x.ndim == 1:
    #     x = x - np.max(x)
    #     x = np.exp(x) / np.sum(np.exp(x))

    return x


#梯度裁減
def clip_grads(grads, max_norm):
    total_norm = 0
    count = 0
    for i in range(len(grads)):
        for grad in grads[i]:
            count += 1
            if count != 6: # 避開c
                total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        count = 0
        for i in range(len(grads)):
            for grad in grads[i]:
                count += 1
                if count != 6: # 避開c
                    grad *= rate


def clip_STgrads(params, grads,st,fix_rate=0.5):
    # 門檻值處理
    c_max = np.max(st)
    c_min = np.min(st)
    # c_std = np.std(st)
    c_ratemax = 0.5
    c_ratemin = 0.5

    # grads[1][2][grads[1][2] > c_std] *= fix_rate
    while (c_ratemax < 1) | (c_ratemin < 1):
        c_norm = np.sum(grads[1][2] ** 2)
        c_ratemax = c_max / (c_norm + 1e-6)
        c_ratemin = abs(c_min / (c_norm + 1e-6))
        if c_ratemax < 1 :
            grads[1][2] *= c_ratemax
        elif c_ratemin < 1:
            grads[1][2] *= c_ratemin
    
    params[1][2][params[1][2] > c_max] *= fix_rate
    params[1][2][params[1][2] < c_min] *= fix_rate
    
    # while grads[1][2][grads[1][2] < c_std].all():
    #     grads[1][2][grads[1][2] > c_std] *= fix_rate

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 訓練資料為one-hot向量時，轉換成正解標籤的索引值
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
