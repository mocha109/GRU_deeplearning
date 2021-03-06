# coding: utf-8
import sys
sys.path.append('..')
# from npTOcp import *
import numpy as np


class SGD:
    '''
    隨機梯度下降法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:
    '''
    Momentum SGD
    '''
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class Nesterov:
    '''
    Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    '''
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i]


class AdaGrad:
    '''
    AdaGrad
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class RMSprop:
    '''
    RMSprop
    '''
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, st_lim, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.st_lim = st_lim
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads, st):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                for p in param:
                    self.m.append(np.zeros_like(p))
                    self.v.append(np.zeros_like(p))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        #st_lr = self.st_lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for i in range(len(params)):
            count = len(params[i])

            for j in range(len(params[i])):
                self.m[j+i*count] += (1 - self.beta1) * (grads[i][j] - self.m[j+i*count])
                self.v[j+i*count] += (1 - self.beta2) * (grads[i][j]**2 - self.v[j+i*count])
                    
                params[i][j] -= lr_t * self.m[j+i*count] / (np.sqrt(self.v[j+i*count]) + 1e-7)
        
        #st更新
        # self.m[5] += (1 - self.beta1) * (grads[1][2] - self.m[5])
        # self.v[5] += (1 - self.beta2) * (grads[1][2]**2 - self.v[5])
        
        params[1][2] = (params[1][2] - np.mean(params[1][2])) / np.std(params[1][2])
        #params[1][2][params[1][2] > np.max(st)] *= self.st_lim
        #params[1][2][params[1][2] < np.min(st)] *= self.st_lim

