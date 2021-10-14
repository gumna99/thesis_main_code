#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import datetime


# In[2]:


import Dataset
#切train test
data = Dataset.Data_set('Training', 'X.csv', 'Y_level3_05s.csv')
#切train test
testdata = Dataset.Data_set('Testing','X.csv', 'Y_level3_05s.csv')
print(testdata.X_n_w.shape,testdata.Y.shape)


# In[3]:


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


# In[10]:


# confusion_matrix
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
 
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):   #plt.cm.Blues、plt.cm.bone_r
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


# ## CNN_model

# In[4]:


from tensorflow.keras.models import load_model
import datetime
from sklearn.model_selection import StratifiedKFold
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores_LSTM = []
trainX_r = trainX.reshape(-1,18)
print(trainX.shape)
for train, test in kfold.split(trainX_r, data.Y):
  # create model

### 建模 EN-CNN
start = time.time()




# load the model from file
sequence_autoencoder = load_model('/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/MODEL_DATA_10_AllNormal_大於600_by_journey/model_3class_05s/model_trained/sequence_autoencoder_100e.h5')
encoder = Model(sequence_autoencoder.input,sequence_autoencoder.layers[-4].output)

# decoder_input = Input(shape=(3,))
# decoder = Model(decoder_input, sequence_autoencoder_b.layers[-1](decoder_input))

encoder.summary()
# decoder.summary()


trainX_encoded = encoder.predict(trainX,batch_size=128)  #
testX_encoded  = encoder.predict(testX,batch_size=128)
# valX_encoded  = encoder.predict(valX,batch_size=128)
print(trainX_encoded.shape)

inputs = Input(shape=(3,3))

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


history_L_E = CNN_model.fit(trainX_encoded, trainY, validation_split=0.1, 
                        epochs=100, batch_size=128, verbose=1,
                           callbacks=[early_stopping,tensorboard_callback])



end = time.time()
print(start, end)
print("執行時間：%f 秒" % (end - start))


# In[5]:


testX_pred_L=CNN_model.predict(testX_encoded)#預測


# In[6]:



acc = CNN_model.evaluate(testX_encoded, testY, batch_size=128, verbose=0)
acc


# In[7]:


def plotLearningCurves(history):
    df = pd.DataFrame(history.history)
    df.plot(figsize=(13,10))  
    plt.legend(markerscale=1000, fontsize=20)
    plt.grid(True) # 顯示網格
    plt.xlabel('epoch', fontsize = 20)  
    plt.ylabel('loss', fontsize = 20) 
    plt.title('Train History')
    plt.gca().set_ylim(0.4, 1)   
    plt.show()
plotLearningCurves(history_L_E)


# In[8]:


from sklearn.metrics import classification_report,confusion_matrix
testX_pred_L = CNN_model.predict(testX_encoded)

y_pred=np.argmax(testX_pred_L, axis=1)
y_test=np.argmax(testY, axis=1)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test,y_pred, digits=4))


# In[11]:



y_pred=np.argmax(testX_pred_L, axis=1)
y_test=np.argmax(testY, axis=1)
cm = confusion_matrix(y_test, y_pred)
class_names = ['1','2','3']
plot_confusion_matrix(cm, class_names)
plot_confusion_matrix(cm, class_names, normalize=False)


# In[ ]:




