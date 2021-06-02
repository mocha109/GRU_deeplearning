from time_layer import *
import pickle
import numpy as np

class Rnngru:
    def __init__(self,st,gamma,var_size, batch_size, time_size,output_size, hidden_size=100):
        N, B, T, H, O = var_size, batch_size, time_size, output_size, hidden_size
        rn = np.random.randn
        
        #初始化權重
        gru_Wx = (rn(T, 4 * H) / np.sqrt(T)).astype('f')
        gru_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        gru_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(B*T, O) / np.sqrt(N)).astype('f')
        affine_b = np.zeros(O).astype('f')
        st_mean = np.mean(st)
        affine_c = np.full((1*(B*T)),st_mean)
        

        #產生各層
        self.layers = [
            TimeGRU(gru_Wx,gru_Wh,gru_b,stateful=True),
            TimeConnection(shape_x=(N, B, T)),
            TimeAffine(affine_W,affine_b,affine_c)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.gru_layer = self.layers[1]

        #把所有權重與梯度整合成清單
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs


    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss 


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



        

