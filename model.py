from time_layer import TimeGRU
from GRU_deeplearning.time_layer import *
import pickle
import numpy as np

class Rnngru:
    def __init__(self, var_size, batch_size, time_size,output_size, hidden_size=100):
        V, B, T, H, O = var_size, batch_size, time_size, output_size, hidden_size
        rn = np.random.randn
        
        #初始化權重
        gru_Wx = (rn(T, 4 * H) / np.sqrt(T)).astype('f')
        gru_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        gru_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, O) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(O).astype('f')

        #產生各層
        self.layers = [
            TimeGRU(gru_Wx,gru_Wh,gru_b,stateful=True),
            TimeConnection(),
            TimeAffine(affine_w,affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        #把所有權重與梯度整合成清單
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    #還在看
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss 

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
