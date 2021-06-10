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
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 訓練資料為one-hot向量時，轉換成正解標籤的索引值
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
