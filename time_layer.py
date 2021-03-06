# from npTOcp import *  # import numpy as np (or import cupy as np)
from pickle import NONE
import numpy as np
from layers import *
from function import softmax, sigmoid_gru, sigmoid_st



class GRU:
    """
    1.此GRU架構下的輸入為各總體經濟變數，計算變數落後期影響力
    2.每一個GRU是在處理單一變數的單一期
    3.輸入xs格式為 N*B*T
    """
    def __init__(self, Wx, Wh, b, gamma):
        
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        self.gamma = gamma

    def forward(self, x, h_prev):
        '''
        本函數使用的sigmoid為變種型，幫助模型適配離散程度不同的資料(加入gamma，但並沒有放入st)
        '''
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        z = sigmoid_gru(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz, self.gamma)
        r = sigmoid_gru(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br, self.gamma)
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
        dbz = np.sum((gamma*dt.T).T, axis=0)         # step z.3.1 : dt * (dsigmoid/dbz)
        dWhz = np.dot(gamma*h_prev.T, dt)      # step z.3.2 : dt * (dsigmoid/dWhz)
        dh_prev += np.dot((gamma*dt.T).T, Whz)       # step z.3.3 : dt * (dsigmoid/dh_prev)
        dWxz = np.dot(gamma*x.T, dt)           # step z.3.4 : dt * (dsigmoid/dWxz)
        dx += np.dot((gamma*dt.T).T, Wxz.T)            # step z.3.5 : dt * (dsigmoid/dx)

        # reset gate(r)
        dr = dhr * h_prev                        # step r.2.1 : dhr * (dhr/dr) ,hr表示r * h_prev
        dt = dr * r * (1-r)                      # step r.2.2 : dr * (dr/dsigmoid) ,sigmoid微分結果在函式註解有
        dbr = np.sum((gamma*dt.T).T, axis=0)         # stpe r.3.1 : dt * (dsigmoid/dbr)
        dWhr = np.dot(gamma*h_prev.T, dt)      # step r.3.2 : dt * (dsigmoid/dWhr)
        dh_prev += np.dot((gamma*dt.T).T, Whr)       # step r.3.3 : dt * (dsigmoid/dh_prev)
        dWxr = np.dot(gamma*x.T, dt)           # step r.3.4 : dt * (dsigmoid/dWxr)
        dx += np.dot((gamma*dt.T).T, Wxr.T)            # step r.3.5 : dt * (dsigmoid/dx)
        
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
    1.每一個timeGRU為一個變數的所有GRU(T=0~T=T)的組合
    2.本model在這部分將所有輸入變數統一處理
    '''
    def __init__(self, Wx, Wh, b, gamma, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful
        self.gamma = gamma

    def forward(self, xs):
        '''
        利用FOR迴圈處理第1期~第t期的gru節點
        '''
        Wx, Wh, b = self.params
        N, B, T = xs.shape  # N為變數數量、B為批次、T為每批次期數
        H = Wh.shape[0]  # Wh.shape[0]的輸出為Wh的列，對應的是xs的T
        self.layers = []
        hs = np.empty((N, B, H), dtype='f')
        
        # 用於處理第一期時，不會有上一期Ht的情況(一般實際執行時self.stateful會設為True)，p.177
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # 將所有期的GRU連結
        for b in range(B):
            layer = GRU(Wx= self.params[0], Wh= self.params[1], b= self.params[2], gamma = self.gamma)  # *參數 : 用於接收實際呼叫函數時，所有多出來的引數會被打包為tuple給該參數
            self.h = layer.forward(xs[:, b, :], self.h)  # xs[:, b, :]的輸出為所有變數的第b批次，結果會另外組成一個新N*T的array
            hs[:, b, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        '''
        反向執行T~0期GRU的backward()並將grads取出
        '''
        Wx, Wh, b = self.params
        N, B, H = dhs.shape
        # N, B, H = dhs.shape
        T = Wx.shape[0]
        gamma = self.gamma

        dxs = np.empty((N, B, T), dtype='f')

        dh = 0
        grads = [0, 0, 0]

        # 由於是反向傳播，因此要將順序顛倒(由後往前_，從第t期GRU往前回朔至第1期GRU
        for b in reversed(range(B)):
            layer = self.layers[b]
            dx, dh = layer.backward(dhs[:, b, :] + dh, gamma)
            dxs[:, b, :] = dx
            
            # 將單一變數的每個gru所產生的Wx、Wh、b分別累計儲存進TimeGRU中forward函式的grads，儲存方式與gru相同
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


class TimeConnection:
    '''
    1.將TimeGRU傳來的hs(N*B*T)轉為BT*N，即每一列為不同變數的同一期
    2.這樣處理是為了在接下來的TimeAffine層中觀察不同變數對於股價報酬率的影響
    '''
    def __init__(self, shape_hs):
        self.N, self.B, self.H = shape_hs
        self.x = None # np.zeros((self.N, self.B * self.H), dtype='f')
        self.dx = None
        self.params = None
        self.grads = None
    
    def forward(self, x):
        '''
        1.先將hs變形為N*BH
        2.利用np.vstack與FOR迴圈將hs重新堆疊變為BH*N
        '''
        N, B, H = self.N, self.B, self.H
        # print(x)

        self.x = x.reshape(N, B * H)
        self.x = self.x.T  # BH*N
        
        # self.x = x[:,0]

        # for t in range(1, B*H):
        #     self.x = np.vstack((self.x, x[:,t]))  # BH*N
        
        return self.x
    
    def backward(self, dx):
        '''
        將dx重新變形為N*B*H
        '''
        N, B, H = self.N, self.B, self.H
        self.dx = dx.T
        self.dx = self.dx.reshape(N, B, H)

        # for n in range(1, N):
        #     self.dx = np.vstack(self.x, dx[:,n])  # N*BH
        #print(self.dx.shape)
        return self.dx



class TimeAffine:
    '''
    本MDOEL中的TimeAffine並非股價本身互乘，而是類似STVAR模型
    '''
    def __init__(self, W, b, c, st, st_gamma):
        self.params = [W, b, c]
        self.st, self.st_gamma = st, st_gamma
        self.grads = [np.zeros_like(W), np.zeros_like(b), np.zeros_like(c)]
        self.x = None

    def forward(self, x):
        BH, N = x.shape
        W, b, c = self.params
        O = int(b.shape[0] / 2)
        Wn, Wst = W[:, :O], W[:, O:2*O]
        bn, bst = b[ :O], b[O:2*O]
        self.transition = sigmoid_st(self.st, self.st_gamma, c)

        #out1 = np.dot(x, Wn) + bn
        out = (np.dot(x, Wn) + bn) + self.transition * (np.dot(x, Wst) + bst)  #BH*O
        #out = out1 + out2
        #out1 = None
        #out2 = None
        self.x = x

        return out

    def backward(self, dout):
        x = self.x
        tran = self.transition
        BH, O = dout.shape
        W, b, c = self.params

        db = np.zeros_like(b)
        dW = np.zeros_like(W)

        db[ :O] = np.sum(dout, axis=0)  # 1*O
        db[O:2*O] = np.sum(tran * dout, axis=0)  # 1*O
        dW[:, :O] = np.dot(x.T, dout)  # N*O
        dW[:, O:2*O] = np.dot((tran*x).T,dout)  # N*O
        dc_temp = (1 / tran) -1
        #dc = np.sum(np.dot(-(tran**2), dc_temp.T) * self.st_gamma, axis=0)  # 1*BH
        dc = -(tran**2) * dc_temp * self.st_gamma  # 1*BH
        dc_temp = None
        dc = dc.T

        dx_temp = np.dot(dout, W[:, :O].T)
        dx = (np.dot(tran * dout, W[:, O:2*O].T) + dx_temp)/2  # BH*N
        dx_temp = None

        self.grads[0][...] = dW
        self.grads[1][...] = db
        self.grads[2][...] = dc
        #print(dx.shape)
        return dx


class TimeSoftmaxWithLoss:
    def __init__(self, batch_size):
        self.params, self.grads = [], []
        self.cache = None
        self.batch_size = batch_size
        self.test = []

    def forward(self, xs, ts):
        BH, O = xs.shape
        B = self.batch_size
        
        ys = softmax(xs)
        # print(ys)
        ls = np.log(ys[ts == 1])
        total_loss = -np.sum(ls)
        # loss = -1 * np.log(ts) * ts
        # total_loss = np.sum(loss)
        avg_loss = total_loss # / BH
        
        # print(xs, ys)

        self.cache = (ts, ys, BH)
        return avg_loss

    def backward(self, dout=1):
        ts, ys, BH = self.cache

        # loss對xs微分
        dx = ys  # 
        dx[ts == 1] = dx[ts == 1] - 1  # 選出dx的所有列中的正確值，並-1(因ts是一[1,0,0...]的矩陣，故最終只會有一個值)
        dx = dx * dout  # 
        # print(dx.shape)
        return dx
