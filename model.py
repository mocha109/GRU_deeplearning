from time_layer import *
import pickle
# from npTOcp import *
import numpy as np

class Rnngru:
    '''
    1.st : 轉換變數(1*BT)，為虛擬變數，用來衡量不同區間下同一變數是否會有不同的影響力
    2.gamma : 標準差(1*N)，為各輸入變數的標準差，用於控制sigmoid形狀
    3.st_gamma : 轉換變數標準差(純值)，功能與gamma相同
    4.
    '''
    def __init__(self, st, gamma, st_gamma, var_size, batch_size, time_size, output_size, hidden_size):

        N, B, T, H, O = var_size, batch_size, time_size, hidden_size, output_size
        rn = np.random.randn
        shape_x=(N, B, H)
        self.st = st  # 用於trainer的梯度裁減
        
        #初始化權重
        gru_Wx = (rn(T, 3 * H)  / np.sqrt(T)).astype('f')
        gru_Wh = (rn(H, 3 * H)  / np.sqrt(H)).astype('f')
        gru_b = np.zeros(3 * H).astype('f')
        affine_W = (rn(N, 2*O)  / np.sqrt(N)).astype('f')
        affine_b = np.zeros(2*O).astype('f')

        st_mean = np.mean(st).astype('f')
        st_std = np.std(st).astype('f')
        #affine_c = np.full(B*T, st_mean).astype('f')
        affine_c = np.random.normal(st_mean, st_std, (1,B*T)).astype('f')
        

        #產生各層
        self.layers = [
            TimeGRU(gru_Wx, gru_Wh, gru_b, gamma, stateful = False),
            TimeConnection(shape_x),
            TimeAffine(affine_W, affine_b, affine_c, st, st_gamma)
        ]
        self.loss_layer = TimeSoftmaxWithLoss(batch_size = B)
        self.gru_layer = self.layers[0]

        #把所有權重與梯度整合成清單
        self.params, self.grads = [], []
        for layer in self.layers:
            if self.layers.index(layer) != 1:
                # self.params += layer.params
                # self.grads += layer.grads
                self.params.append(layer.params)
                self.grads.append(layer.grads)


    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs


    def forward(self, xs, ts):
        score = self.predict(xs)
        avg_loss = self.loss_layer.forward(score, ts)
        return avg_loss


    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


    def reset_state(self):
        self.gru_layer.reset_state()


    def save_params(self,file_name='gru.pkl'):
        with open(file_name,'wb')as f:
            pickle.dump(self.params,f)
    

    def load_params(self,file_name='gru.pkl'):
        with open(file_name,'rb') as f:
            self.params=pickle.load(f)



        

