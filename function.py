from npTOcp import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gru(x,gamma):
    # 此GAMMA為一1*N矩陣
    return 1 / (1 + np.exp(-gamma.T * x))

def sigmoid_st(st,gamma,c):
    # 此一GAMMA為純值
    return 1 / (1 + np.exp(-gamma*(st-c))) 


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


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 訓練資料為one-hot向量時，轉換成正解標籤的索引值
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  