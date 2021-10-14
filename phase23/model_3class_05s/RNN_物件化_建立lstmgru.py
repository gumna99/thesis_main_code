#!/usr/bin/env python
# coding: utf-8

# ## 物件化管理
# ### RNN、GRU、SimpleRNN
# ### Y做正規化
# 

# In[4]:


import random
import datetime
import time
import pickle
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Model   # 按順序建立的神經網路
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D, LSTM, TimeDistributed,SimpleRNN # Dense全連線層
from tensorflow.keras.layers import RepeatVector,Input, Conv1D, Lambda, GRU, MaxPooling1D, multiply
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error
# confusion_matrix
from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import confusion_matrix


# In[2]:


class RNN_model_3C:        
    def __init__(self, rnn_type, L1_num, L2_num, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.L1_num = L1_num
        self.L2_num = L2_num
        self.cm = None
        
        self.inputs = Input(shape=(3, 6))
        if rnn_type == "LSTM":
            self.lstm1 = LSTM(self.L1_num, activation='relu',return_sequences=True)(self.inputs)
            self.D1=Dropout(0.2)(self.lstm1)
            self.lstm2 = LSTM(self.L2_num, activation='relu',return_sequences=False)(self.D1)
            self.D2=Dropout(0.2)(self.lstm2)
            self.regression_layers= Dense(3, activation='softmax')(self.D2)
            
        if rnn_type == "GRU":
            self.GRU1 = GRU(self.L1_num, activation='relu',return_sequences=True)(self.inputs)
            self.D1=Dropout(0.2)(self.GRU1)
            self.GRU2 = LSTM(self.L2_num, activation='relu',return_sequences=False)(self.D1)
            self.D2=Dropout(0.2)(self.GRU2)
            self.regression_layers= Dense(3, activation='softmax')(self.D2)
            
        if rnn_type == "SimpleRNN":
            self.SimpleRNN1 = SimpleRNN(self.L1_num, activation='relu',return_sequences=True)(self.inputs)
            self.D1=Dropout(0.2)(self.SimpleRNN1)
            self.SimpleRNN2 = SimpleRNN(self.L2_num, activation='relu',return_sequences=False)(self.D1)
            self.D2=Dropout(0.2)(self.SimpleRNN2)
            self.regression_layers= Dense(3, activation='softmax')(self.D2)
            
        self.model = Model(self.inputs, self.regression_layers)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2) 
        self.model.summary()    
        
    def training(self, trainX, trainY):
        start = time.time()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        history = self.model.fit(trainX, trainY, validation_split=0.1,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs, verbose=1,
                                 callbacks=[self.early_stopping])
        end = time.time()
        print(start, end)
        print("執行時間：%f 秒" % (end - start))
        return history
        
    def plotLearningCurves2(self, history):
        df = pd.DataFrame(history.history)
        df.plot(figsize=(13,10))  
        plt.legend(markerscale=1000, fontsize=20)
        plt.grid(True) # 顯示網格
        plt.xlabel('epoch', fontsize = 20)  
        plt.ylabel('loss', fontsize = 20) 
        plt.title('Train History')
        plt.gca().set_ylim(0.2, 1)   
        plt.show()
        
    def make_predict(self, testX):     #不用inverse Y
        testX_pred=self.model.predict(testX)#預測testX
        return  testX_pred
            #       EX:  testX_pred = make_predict(data.testX)

    def eval(self, testX, testy):    
        accuracy = self.model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)         
        _, acc = self.model.evaluate(data.testX, testY, batch_size=128, verbose=0)
        return acc


    def plot_cm(self,testY, testX_pred_i_LSTM):
        from sklearn.metrics import classification_report,confusion_matrix
        y_pred=np.argmax(testX_pred_i_LSTM, axis=1)
        y_test=np.argmax(testY, axis=1)
        cm = confusion_matrix(y_test, y_pred)

        print(cm)
        print(classification_report(y_test,y_pred,digits=4))
        y_pred=np.argmax(testX_pred_i_LSTM, axis=1)
        y_test=np.argmax(testY, axis=1)
        cm = confusion_matrix(y_test, y_pred)

        class_names = ['1','2','3']
        plot_confusion_matrix(cm, class_names)
        plot_confusion_matrix(cm, class_names, normalize=False)
    
def plot_confusion_matrix( cm, classes=['1','2','3'],
                      normalize=True,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):   #plt.cm.Blues、plt.cm.bone_r
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import classification_report
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize = (7, 7))
    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title,size=20)
    plt.colorbar(aspect=5)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=18)
    plt.yticks(tick_marks, classes, size=18)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=18,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',size=20)
    plt.xlabel('Predicted label',size=20)
    plt.tight_layout()

    


# In[ ]:





# # 資料讀取建立
# ## X>>前normal過， Y_level3

# In[27]:


import Dataset
#切train test# data.dataset_split(0.2,0.1,42)

data = Dataset.Data_set('Training', 'X.csv', 'Y_level3_05s.csv')
print(data.X_n_w.shape,data.Y.shape)

testdata = Dataset.Data_set('Testing','X.csv', 'Y_level3_05s.csv')
print(testdata.X_n_w.shape,testdata.Y.shape)


# In[64]:


pd.DataFrame(trainX.reshape(-1,18)) #檢視


# In[28]:


#作one-hotincoding
from  keras.utils import to_categorical
# zero-offset class values  從0開始計類
trainX = data.X_n_w
trainY = data.Y - 1
# one hot encode y
trainY = to_categorical(trainY)
print(trainX.shape, trainY.shape)

testX = testdata.X_n_w
testY = testdata.Y - 1
# one hot encode y
testY = to_categorical(testY)
print(testX.shape, testY.shape)


# In[29]:


# 檢視比例

unique, counts = np.unique(data.Y, return_counts=True)
print(dict(zip(unique, counts)));


# # LSTM 模型

# In[51]:


LSTM_model = RNN_model_3C("LSTM", 64, 16, 128, 100)  
#training
train_history_LSTM = LSTM_model.training(trainX, trainY)


# In[57]:


# acc = LSTM_model.eval(data.testX, testY)
# print('acc: ', acc)
# #畫出學習曲線
LSTM_model.plotLearningCurves2(train_history_LSTM)
# 預測pred
testX_pred_i_LSTM = LSTM_model.make_predict(testX)
# # plot cm
LSTM_model.plot_cm(testY, testX_pred_i_LSTM)


# In[59]:


LSTM_model.model.save('trained_model/LSTM_model.h5')


# ## GRU模型

# In[18]:


GRU_model = RNN_model_3C("GRU", 32, 16, 128, 100)  
#training
train_history_GRU = GRU_model.training(trainX, trainY)


# In[19]:


# acc = GRU_model.eval(testX, testY)
# print('acc: ', acc)
#畫出學習曲線
GRU_model.plotLearningCurves2(train_history_GRU)
# 預測pred
testX_pred_i_GRU = GRU_model.make_predict(testX)
# plot cm
GRU_model.plot_cm(testY, testX_pred_i_GRU)


# In[ ]:


# GRU_model.model.save('trained_model/GRU_model.h5') 


# # SIMPLE_RNN

# In[15]:


SIMPLE_model = RNN_model_3C("SimpleRNN", 32, 16, 128, 100) 
#training|
train_history_SIM = SIMPLE_model.training(trainX, trainY)


# In[17]:


# acc = SIMPLE_model.eval(data.testX, testY)
# print('acc: ', acc)
# #畫出學習曲線
SIMPLE_model.plotLearningCurves2(train_history_SIM)
# 預測pred
testX_pred_i_SIM = SIMPLE_model.make_predict( testX)
# plot cm
SIMPLE_model.plot_cm(testY, testX_pred_i_GRU)


# In[ ]:


# SIMPLE_model.model.save('trained_model/SIMPLE_model.h5') 


# # CNN

# In[70]:


from tensorflow.keras.models import load_model
import datetime
### 建模 EN-LSTM
start = time.time()


inputs = Input(shape=(3,6))

c1 = Conv1D(16,1 , activation="relu")(inputs)
c1 = Conv1D(8,1 , activation="relu")(c1)
p1 = MaxPooling1D(pool_size=2)(c1)

f = Flatten()(p1)
d = Dense(8, activation="relu")(f)
regression_layers = Dense(3, activation='softmax')(d)
CNN_model = Model(inputs, regression_layers)
CNN_model.compile(loss="categorical_crossentropy", optimizer="adam")
CNN_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2) 
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


history_C_E = CNN_model.fit(trainX, trainY, validation_split=0.1, 
                        epochs=100, batch_size=128, verbose=1,
                           callbacks=[early_stopping,tensorboard_callback])



end = time.time()
print(start, end)
print("執行時間：%f 秒" % (end - start))


# In[71]:


test_pred_CNN = CNN_model.predict(testX)

def plot_cm(Y, testX_pred_i_LSTM):
    from sklearn.metrics import classification_report,confusion_matrix
    y_pred=np.argmax(testX_pred_i_LSTM, axis=1)
    y_test=np.argmax(Y, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    print(classification_report(y_test,y_pred,digits=4))

    class_names = ['1','2','3']
    plot_confusion_matrix(cm, class_names)
    plot_confusion_matrix(cm, class_names, normalize=False)
    
plot_cm(testY, test_pred_CNN)


# # RF

# In[72]:


from sklearn.ensemble import RandomForestClassifier


start = time.time()


RF_model = RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)

# Y = data.trainY.reshape(-1)
trainX_r = trainX.reshape(-1,18)
testX_r = testX.reshape(-1,18)

history_RF = RF_model.fit(trainX_r,trainY )#,random_state=42

testX_pred_L=RF_model.predict(testX_r)#預測



end = time.time()
print(start, end)
print("執行時間：%f 秒" % (end - start))

from sklearn.metrics import classification_report,confusion_matrix

y_pred=np.argmax(testX_pred_L, axis=1)
y_test=np.argmax(testY, axis=1)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test,y_pred, digits=4))


# In[73]:


def plot_cm(testY, testX_pred_i_LSTM):
    from sklearn.metrics import classification_report,confusion_matrix
#     y_pred=np.argmax(testX_pred_i_LSTM, axis=1)
#     y_test=np.argmax(testY, axis=1)
    cm = confusion_matrix(testY, testX_pred_i_LSTM)

    print(cm)
    print(classification_report(testY,testX_pred_i_LSTM,digits=4))
#     y_pred=np.argmax(testX_pred_i_LSTM, axis=1)
#     y_test=np.argmax(testY, axis=1)
#     cm = confusion_matrix(y_test, y_pred)

    class_names = ['1','2','3']
    plot_confusion_matrix(cm, class_names)
    plot_confusion_matrix(cm, class_names, normalize=False)
    
plot_cm(y_test, y_pred)


# In[75]:


import joblib

joblib.dump(RF_model, 'trained_model/RF_model_fromrnn')


# In[ ]:




