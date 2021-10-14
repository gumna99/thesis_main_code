#!/usr/bin/env python
# coding: utf-8

# 從DATA_10/x_10_600.csv 找出前10駕駛超過600秒的屢次，各個欄位的平均值、標準差，給未來SAX轉換中的標準化使用。
# 
# 從data_中山_bydriver/ 各個駕駛的屢次  找出大於600的X值，進行處理：
#     1. 儲存原始x
#     2. data_10_600 
#     
# 製作XY、進行label 切分

# In[19]:


#複製檔案
import pandas as pd
import time
import os
import pickle
import SCORE
from shutil import copyfile


start = time.time()

Bdriverid=[31932651,31931745,31929871,31932404,31902662,
           31932370,31928725,31932602,31931992,31902118]    
for driver in Bdriverid:
    print('正在執行駕駛: ', driver ,'...')
#     pickle_list = glob.glob(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/*.pickle')#回傳檔案的路近
    pickle_list = os.listdir(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/') #回傳檔案名+副檔名
    print('本駕駛有幾趟旅次: ', len(pickle_list))
    
    _count=0 # 計本駕駛超過600的次數
    for p in pickle_list:
        print('旅次名稱: ', p)
        file_name = p.split(".")[0] #分開檔名與副檔名
        with open (f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/{p}', 'rb') as f:
            d=pickle.load(f)
        print('本旅次有幾筆: ', len(d['time_data']))
        # 每旅次抓出time_data
        if len(d['time_data'])>600 :
        
            #新增每個旅次的資料夾、並且複製pickle檔案
            os.mkdir(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}")
            copyfile(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/{p}',
                     f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/{p}")


# # 先組成全部X的csv，為了以後取得各col的mean std使用
# ## 得到X_all.scv 
# 

# In[5]:


start = time.time()


df_x = pd.DataFrame(columns=['_time', 'x', 'y','z','yaw','roll','pitch','car_id','journey_id'] )
df_y = pd.DataFrame(columns=['score_3s_all'])
df_count=0
X=np.empty(shape=(0,3,6))

for driver in Bdriverid:
    print('正在執行駕駛: ', driver ,'...')
#     pickle_list = glob.glob(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/*.pickle')#回傳檔案的路近
    pickle_list = os.listdir(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/') #回傳檔案名+副檔名
    print('本駕駛有幾趟旅次: ', len(pickle_list))
    
    _count=0 # 計本駕駛超過600的次數
    for p in pickle_list:
        print('旅次名稱: ', p)
        file_name = p.split(".")[0] #分開檔名與副檔名
        with open (f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{p}', 'rb') as f:
            d=pickle.load(f)
        print('本旅次有幾筆: ', len(d['time_data']))
        # 抓出time_data
        if len(d['time_data'])>600:
            df_t = pd.DataFrame(d['time_data'])
#             df_s = pd.DataFrame(d['score_3s_all'],columns=['score_3s_all'])
            df_t['journey_id']=file_name

            df_x=df_x.append(df_t) #儲存原始的timedata
#             df_y=df_y.append(df_s) #儲存
            _count+=1
            df_count+=1
        # 抓出score_3s_all
    print(f'本駕駛有{_count}筆的資料 .')
    print('\n')
print('---------------------------------------')
print(f'總共有{df_count}筆超過600的資料.')
print(f'總共有{len(df_x)}筆資料x')
# print(f'總共有{len(df_x)}筆資料x, 總共有{len(df_y)}筆資料y ........')
# print(f'總共有{len(X)}筆切割過的資料X, 總共有{len(df_y)}筆資料y ........')

end = time.time()
print(start, end)
print("執行時間：%f 秒" % (end - start))

df_x.to_csv('DATA_All/X_all.csv', index=False)


# In[2]:


import pandas as pd
import numpy as np


# In[ ]:


# 取得各col的mean std
def get_meanstd(col):
    # load 全部原始資料
    X = pd.read_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/DATA_All/X_all.csv")
    #分別取得個軸的mean std
    if col == 'x':
        mean = np.nanmean(X.x)
        std = np.nanstd(X.x)
        return mean, std
    if col == 'y':
        mean = np.nanmean(X.y)
        std = np.nanstd(X.y)
        return mean, std        
    if col == 'z':    
        mean = np.nanmean(X.z)
        std = np.nanstd(X.z)
        return mean, std
    if col == 'yaw':
        mean = np.nanmean(X.yaw)
        std = np.nanstd(X.yaw)
        return mean, std
    if col == 'roll':
        mean = np.nanmean(X.roll)
        std = np.nanstd(X.roll)
        return mean, std
    if col == 'pitch':
        mean = np.nanmean(X.pitch)
        std = np.nanstd(X.pitch)
        return mean, std
# 正規
def normalize(col, ts):
    mean, std = get_meanstd(col)
    return (ts - mean) / std

from sklearn.preprocessing import MinMaxScaler, StandardScaler                               

# 正規+切window
def Normalized_window_sliding(data, window_size, scaler_type ,scale=True):
    #先對X做正規化+平移窗格
    _l = len(data) 
    data = scaler_type().fit_transform(data)
    Xs = []
    for i in range(0, (_l - window_size+1)):
    # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
    print(f'資料形狀: {len(Xs)}')
    return (Xs)
# 切window
def window_sliding(data, window_size):
    #先對X做正規化+平移窗格
    _l = len(data) 
    Xs = []
    for i in range(0, (_l - window_size+1)):
    # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
    print(f'資料形狀: {len(Xs)}')
    return (Xs)

#按照不同閾值切label
def label_level_3_05(value):
    if value < (mean - 0.5* std):
        return str(1)
    elif value > (mean - 0.5* std) and  value < (mean + 0.5* std):
        return str(2)
    else:
        return str(3)
    
def label_level_3_1(value):
    if value < (mean - 1* std):
        return str(1)
    elif value > (mean - 1* std) and  value < (mean + 1* std):
        return str(2)
    else:
        return str(3) 
    
def label_level_3_15(value):
    if value < (mean - 1.5* std):
        return str(1)
    elif value > (mean - 1.5* std) and  value < (mean + 1.5* std):
        return str(2)
    else:
        return str(3)


# # main
# # 對每趟旅次資料夾的資料作處理、LABEL

# In[18]:


# --------------main--------------
# -------------做score------------

import pandas as pd
import time
import os
import pickle
import SCORE
from shutil import copyfile

start = time.time()
################
Bdriverid=[31932651,31931745,31929871,31932404,31902662,
           31932370,31928725,31932602,31931992,31902118]            

X_count=0
X_w_count=0
# Y_count=0
Y_w_count=0
df_count=0
# X_N_W=np.empty(shape=(0,3,6))  #用來放以切之X

for driver in Bdriverid:
    print('正在執行駕駛: ', driver ,'...')
#     pickle_list = glob.glob(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/{driver}/*.pickle')#回傳檔案的路近
    pickle_list = os.listdir(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/') #回傳檔案名+副檔名
    print('本駕駛有幾趟旅次: ', len(pickle_list))
    
    _count=0 
    for p in pickle_list:
        print('旅次名稱: ', p)
        file_name = p.split(".")[0] #分開檔名與副檔名
        with open (f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{p}/{p}.pickle', 'rb') as f:
            d=pickle.load(f)
        print('本旅次有幾筆: ', len(d['time_data']))
        # 每旅次抓出time_data
        if len(d['time_data'])>600 :
            
#             df_x = pd.DataFrame(columns=['_time', 'x', 'y','z','yaw','roll','pitch','car_id','journey_id'] )
            df_y = pd.DataFrame(columns=['score_3s_x','score_3s_y','score_3s_z','score_3s_yaw','score_3s_roll',
                             'score_3s_pitch','score_3s_all'])

            df_XN =  pd.DataFrame(columns=['x', 'y','z','yaw','roll','pitch','car_id','journey_id'] )

            X_N_W=np.empty(shape=(0,3,6))  #用來放以切之X

            print('--main----go------')
            df_t = pd.DataFrame(d['time_data'])

            df_xn = pd.DataFrame() #紀載個軸的x
            df_s = pd.DataFrame() #放切割後 每軸的分數                  
            x_n = []  # 存放每一軸
            column = ['x','y','z','yaw','roll','pitch']
            for col in column:
                ts = normalize(col , df_t[col])
                
                df_xn[col] = ts  #紀載個軸的x
                
                ##### 計算scoring
                x1 = SCORE.SAX_trans(ts=ts,w=len(df_t[col]),alpha=7)  #讀進物件
                x2 = x1.to_sax2()
                #SAX是一整串字串 ex:'123456789',進行分割
                x2_splited = []
                for s in range(len(x2)-2):
                    x2_splited.append(x2[s:s+3])
#                 print('score_3s'+col+'未來的個數: ', len(x2_splited)) #應該要跟len(data_3s)一樣
                #做分數
                score = SCORE.sax2_to_score.score_3s(x2_splited)
        
                # 存在dataframe
                df_s['score_3s_'+ col ] = score
            #計算 score_3s_all
            df_s['score_3s_all'] = df_s['score_3s_x']+df_s['score_3s_y']+df_s['score_3s_z']+df_s['score_3s_yaw']+df_s['score_3s_roll']+df_s['score_3s_pitch']                
            #放入Y
            df_y = df_y.append(df_s)    #儲存儲存score            
            df_y.to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/score.csv", index=False)
            
            #3class切分
            #get 母體Y的平均標準差>>給labeling使用
            Y = pd.read_csv("/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/DATA_All/score.csv")
            score_3s_all = Y["score_3s_all"]
            mean = np.mean(score_3s_all.values)
            std = np.std(score_3s_all.values)
            print('母體Y的平均、標準差: ', mean, std)
            
            #取旅次的六軸分數  做labeling
            df_y['level_3_05s'] = df_y.score_3s_all.apply(label_level_3_05)
            df_y['level_3_1s'] = df_y.score_3s_all.apply(label_level_3_1)
            df_y['level_3_15s'] = df_y.score_3s_all.apply(label_level_3_15)
            print("本旅次Y值分類(df_y['level_3_05s'])的shape: ", df_y['level_3_05s'].shape)
            print("本旅次Y值分類(df_y['score_3s_all'])的shape: ", df_y['score_3s_all'].shape)
            #先儲存
            df_y["level_3_05s"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_05s.csv", index=False)
            df_y["level_3_1s"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_1s.csv", index=False)
            df_y["level_3_15s"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_15s.csv", index=False)


            #做y軸的分數
            score_3s_y = Y["score_3s_y"]
            mean = np.mean(score_3s_y.values)
            std = np.std(score_3s_y.values)
            print('母體Y的y軸平均、標準差: ', mean, std)
            
            #取旅次的Y值 做labeling
            df_y['level_3_05s_y'] = df_y.score_3s_y.apply(label_level_3_05)
            df_y['level_3_1s_y'] = df_y.score_3s_y.apply(label_level_3_1)
            df_y['level_3_15s_y'] = df_y.score_3s_y.apply(label_level_3_15)
            print("本旅次Y值分類(df_y['level_3_05s_y'])的shape: ", df_y['level_3_05s_y'].shape)
            print("本旅次Y值分類(df_y['score_3s_all'])的shape: ", df_y['score_3s_all'].shape)
            #先儲存
            df_y["level_3_05s_y"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_05s_y.csv", index=False)
            df_y["level_3_1s_y"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_1s_y.csv", index=False)
            df_y["level_3_15s_y"].to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/Y_level3_15s_y.csv", index=False)
            
            
            
            # 處理X，進行視窗切割
            x_nw = window_sliding(df_xn, 3)
            x_nw = np.array(x_nw)
            X_N_W = np.concatenate([X_N_W,x_nw])
            print(f'切割後的X形狀: {X_N_W.shape}')

            
            df_xn['journey_id']=file_name  #加入journey
            df_xn['car_id'] = df_t['car_id']
            df_XN = df_XN.append(df_xn)  #儲存timedata正規化後加上屢次
            


            _count+=1 #算本駕駛超過600旅次數
            df_count+=1 #算總共超過600旅次數
            
            
            print('----------本旅次分數、分類 計算完畢----------------') 

            
            df_y.to_csv(f"/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/score.csv", index=False)
            df_XN.to_csv(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/X_n.csv', index=False)
            pd.DataFrame(X_N_W.reshape(-1,18)).to_csv(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/X.csv', index=False)
            print('----------本旅次儲存完畢----------------')   
            print(f'總共有{len(df_XN)}筆資料x')
            print(f'總共有{len(X_N_W)}筆切割過的資料X, 總共有{len(df_y)}筆資料y ........')
            print(f'切割後的X形狀: {X_N_W.shape}, 切割後的Y形狀: {df_y.shape} ......') 
            # 只取X_y
            XR = X_N_W.reshape(-1,6)
            XRR = pd.DataFrame(XR[:,1])
            XRR.to_csv(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/X_y.csv', index=False)           
            print(f'總共有{len(XRR)}筆切割過的資料X_y, 總共有{len(df_y)}筆資料y ........')
            #統計
            sta={}
            sta['旅次筆數'] = len(d['time_data'])
            sta['X_N_W筆數'] = len(X_N_W)       
                                 
            unique, counts = np.unique(df_y["level_3_05s"], return_counts=True)
            sta['05s類別筆數'] = counts
            unique, counts = np.unique(df_y["level_3_1s"], return_counts=True)
            sta['1s類別筆數'] = counts
            unique, counts = np.unique(df_y["level_3_15s"], return_counts=True)
            sta['15s類別筆數'] = counts
            np.save(f'/3T_HD/Neng/智慧機車_data/data_中山_bydriver/Best_10_drivers/DATA_10_AllNormal_大於600_by_journey/{driver}/{file_name}/stastistic.npy', sta)              
                        
            
            X_count += len(d['time_data'])
            X_w_count += len(X_N_W)
            Y_w_count += len(df_y)
            print('下一個旅程 ...')
    
    print(f'本駕駛有{_count}筆>600的資料 .')
    print("下一個駕駛 ...")
    print('\n')
print('---------------------------------------')
print(f'總共有{df_count}筆>600的資料.')
print(f'總共有{X_count}筆x的資料.')
print(f'總共有{X_w_count}筆切割窗格過的x資料.')


end = time.time()
print(start, end)
print("執行時間：%f 秒" % (end - start))

#X_N_W  做X
#df_XN 儲存 X正規化後的資料
# df_y儲存各columns 的分數、加上分級
# Y 做Y


# In[ ]:




