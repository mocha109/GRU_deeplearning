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