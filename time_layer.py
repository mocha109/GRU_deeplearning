from GRU_deeplearning.np import *  # import numpy as np (or import cupy as np)
from GRU_deeplearning.layers import *
from GRU_deeplearning.functions import softmax, sigmoid_gru

#jerry請看這，我在想TimeEmbedding的W是否形狀不應和gru相同，一開始輸入的資料應該是T*D(T是期數、D是同一總經變數的不同形式)
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad

        return None

class GRU:
    """
    1.此類別為處理各總體經濟變數，計算變數落後期影響力
    2.每一個GRU是在處理單一變數的單一期
    """
    def __init__(self, Wx, Wh, b):
        
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        '''
        本函數使用的sigmoid為變種型，加入gamma
        '''
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

    def backward(self, dh_next, gamma):
        """
        sigmoid和tanh函數微分結果
        1.z = sigmoid(x) -> dz/dx = z * (1-z)
        2.t = tanh(x)    -> dt/dx = 1 - t**2
        """
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat =dh_next * z                # step 1.1 : dh_next * (dh_next/dh_hat)
        dh_prev = dh_next * (1-z)          # step 1.2 : dh_next * (dh_next/dh_prev)

        # tanh(h)
        dt = dh_hat * (1 - h_hat ** 2)     # step h.2.1 : dh_hat * (dh/dtanh) ,tanh微分結果在函式註解有
        dbh = np.sum(dt, axis=0)           # step h.3.1 : dt * (dtanh/dbh) ,axis=0表示沿縱向加總
        dWhh = np.dot((r * h_prev).T, dt)  # step h.3.2 : dt * (dtanh/dWh)
        dWxh = np.dot(x.T, dt)             # step h.3.3 : dt * (dtanh/dWxh)
        dx = np.dot(dt, Wxh.T)             # step h.3.4 : dt * (dtanh/dx)
        dhr = np.dot(dt, Whh.T)            # step h.4.1 : dt * (dtanh/dhr) ,hr表示r * h_prev
        dh_prev += r * dhr                 # step h.4.2 : dhr * (dhr/dh_prev)

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev  # step z.2.1 : dh_next * (dh_next/dz)
        dt = dz * z * (1-z)                      # step z.2.2 : dz * (dz/dsigmoid) ,sigmoid微分結果在函式註解有
        dbz = np.sum(dt, axis=0)*gamma           # step z.3.1 : dt * (dsigmoid/dbz)
        dWhz = np.dot(h_prev.T, dt)*gamma        # step z.3.2 : dt * (dsigmoid/dWhz)
        dh_prev += np.dot(dt, Whz.T)*gamma       # step z.3.3 : dt * (dsigmoid/dh_prev)
        dWxz = np.dot(x.T, dt)*gamma             # step z.3.4 : dt * (dsigmoid/dWxz)
        dx += np.dot(dt, Wxz.T)*gamma            # step z.3.5 : dt * (dsigmoid/dx)

        # reset gate(r)
        dr = dhr * h_prev                        # step r.2.1 : dhr * (dhr/dr) ,hr表示r * h_prev
        dt = dr * r * (1-r)                      # step r.2.2 : dr * (dr/dsigmoid) ,sigmoid微分結果在函式註解有
        dbr = np.sum(dt, axis=0)*gamma           # stpe r.3.1 : dt * (dsigmoid/dbr)
        dWhr = np.dot(h_prev.T, dt)*gamma        # step r.3.2 : dt * (dsigmoid/dWhr)
        dh_prev += np.dot(dt, Whr.T)*gamma       # step r.3.3 : dt * (dsigmoid/dh_prev)
        dWxr = np.dot(x.T, dt)*gamma             # step r.3.4 : dt * (dsigmoid/dWxr)
        dx += np.dot(dt, Wxr.T)*gamma            # step r.3.5 : dt * (dsigmoid/dx)
        
        #將權重與截距儲存起來以便於修正
        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))
        
        #grads中的0項專用於儲存Wx、1項專用於儲存Wh、2項專用於儲存b
        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev

class TimeGRU:
    '''
    每一個timeGRU為一個變數的所有GRU(T=0~T=T)的組合
    '''
    def __init__(self, Wx, Wh, b, stateful=True):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        '''
        利用迴圈處理第1期~第t期的gru節點
        '''
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # N為批次、T為期數、D為維度
        H = Wh.shape[0]  # Wh.shape[0]的輸出為Wh的列，對應的是xs的D
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        
        # 用於處理第一期時，不會有上一期Ht的情況(一般實際執行時self.stateful會設為True)，p.177
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # 將所有期的GRU連結
        for t in range(T):
            layer = GRU(*self.params)  # *參數 : 用於接收實際呼叫函數時，所有多出來的引數會被打包為tuple給該參數
            self.h = layer.forward(xs[:, t, :], self.h)  # xs[:, t, :]的輸出為所有批次、維度的第t列，結果會另外組成一個新的array
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        '''
        1
        '''
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')

        dh = 0
        grads = [0, 0, 0]

        # 由於是反向傳播，因此要將順序顛倒(由後往前_，從第t期GRU往前回朔至第1期GRU
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            
            # 將單一變數的每個gru所產生的Wx、Wh、b分別累計儲存進grads，儲存方式與gru相同
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # enumerate(sequence, [start=0])會產生一個tuple組成的list，tuple內容為start起始的索引值和sequence內元素
        # 以下方為例，i會取出索引值；grad會取出對應的sequence元素
        # 將grads內的Wx、Wh、b存入類別TimeGRU的grads中
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None