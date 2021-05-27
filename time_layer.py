from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid_gru

class GRU:
    def __init__(self, Wx, Wh, b):
        
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        z = sigmoid_gru(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid_gru(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bh)
        h_next = (1-z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next,gamma):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat =dh_next * z            #h_next=ht in p348
        dh_prev = dh_next * (1-z)      

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dbh = np.sum(dt, axis=0)
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1-z)
        dbz = np.sum(dt, axis=0)*gamma
        dWhz = np.dot(h_prev.T, dt)*gamma
        dh_prev += np.dot(dt, Whz.T)*gamma
        dWxz = np.dot(x.T, dt)*gamma
        dx += np.dot(dt, Wxz.T)*gamma    #2021/05/27 22:06

        # reset gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev