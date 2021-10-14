#!/usr/bin/env python
# for classification


# coding: utf-8



# In[ ]:


import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras import Sequential, Model   # 按順序建立的神經網路
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D, LSTM, TimeDistributed # Dense全連線層
from tensorflow.keras.layers import RepeatVector, Dense, Flatten, Input, Dropout, Conv1D, Lambda, GRU, Softmax, MaxPooling1D, multiply
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping


class Data_set:
    def __init__(self, X, Y):
        self.X = pd.read_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal/{X}")
        self.Y = pd.read_csv(f"/3T_HD/Neng//智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal/{Y}")
        self.X = self.X.values
        self.Y = self.Y.values
        self.scaler_x = MinMaxScaler()
        self.X_n = self.scaler_x.fit_transform(self.X)
        self.X_n_w  = self.X_n.reshape(-1,3,6)  #需要改成3,6  ###備註:已經做過NORMALIZE 和 切割視窗
        
        
        
        self.trainX=None
        self.trainY=None
        self.testX=None
        self.testY=None
        self.valX=None
        self.valY=None

    
    def dataset_split(self,split_size_ts, split_size_va,seed ):    
        tr_x_0, ts_x, tr_y_0, ts_y = [np.array(x) for x in train_test_split(self.X_n_w, self.Y,
                                                                  test_size=split_size_ts, 
                                                                            random_state=seed,
                                                                           stratify = self.Y)]
    #   assert tr_x.shape[2] == ts_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))
        tr_x, va_x, tr_y, va_y = [np.array(x) for x in train_test_split(tr_x_0, tr_y_0,
                                                                  test_size=split_size_va, random_state=seed
                                                                       ,stratify = tr_y_0)]
        tr_y = tr_y.reshape(-1,1)
        ts_y = ts_y.reshape(-1,1)
        va_y = va_y.reshape(-1,1)
        print(f'trainX形狀: {tr_x.shape} ,testX形狀: {ts_x.shape}, vaX形狀: {va_x.shape},trainY形狀: {tr_y.shape}, testY形狀: {ts_y.shape},  vaY形狀: {va_y.shape}')
        self.trainX=tr_x
        self.trainY=tr_y
        self.testX=ts_x
        self.testY=ts_y
        self.valX=va_x
        self.valY=va_y

    def Normalized_window_sliding(data, window_size, scaler_type=StandardScaler,scale=True):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler                               
        #先對X做正規化+平移窗格
        _l = len(data) 
        data = scaler_type().fit_transform(data)
        Xs = []
        for i in range(0, (_l - window_size+1)):
        # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
            Xs.append(data[i:i+window_size])
        print(f'資料形狀: {len(Xs)}')
        return (Xs)

