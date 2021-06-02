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
            TimeGRU()
        ]
