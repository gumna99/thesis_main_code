# 以深度學習分析機車駕駛風險指跡應用於駕駛行為車險服務
## Analyzing Motorcyclist's Risk-aware Fingerprint for the Usage-based Insurance by Using Deep Learning
- 碩士論文之主要程式碼。
- 僅節錄重要程式碼、模型，不包含實驗等過程。
- 論文待審查發表，目前不公開






## 主要功能
### phase1
- SCORE.py
    - SAX編碼以及計算分數
    - SAX_trans - 轉成編碼
    - sax2_to_score - 轉成分數
    - 注意：
        - 需要先把此py，放到與code同一層路徑
    - 使用方式
        
        ```python
        x1 = SCORE.SAX_trans(ts=ts,w=len(df_t[col]),alpha=7)  #ts- 時間序列 w- 間隔 #alpha - 分幾級
        
        x2 = x1.to_sax2()
        
        #SAX是一整串字串 ex:'123456789',進行分割3秒一窗格
        x2_splited = []
        for s in range(len(x2)-2):
        x2_splited.append(x2[s:s+3])
        print('score_3s'+col+'未來的個數: ', len(x2_splited)) #應該要跟len(data_3s)一樣
        
        #做分數
        score = SCORE.sax2_to_score.score_3s(x2_splited)
        ```
- Dataset.py
    - 用於建模前讀取數據(物件導向方式儲存input output)
    - 內容：讀取、利用Minmax正規、再把檔案分成train test validation
    - 注意：
        - 30 31行，讀取檔案改成你要的檔案路徑
        - 需要先把此py，放到與code同一層路徑
    - 使用
        
        ```python
        import Dataset
        #切train test
        data = Dataset.Data_set('X.csv', 'Y_level3_05s.csv')
        # data.dataset_split(0.2,0.1,42)
        print(data.X_n_w.shape,data.Y.shape)
        #打data.___ 即可得到需要的數據
        ```
        
- DATA_10_scoring_with6_by_3second_中山_ALLNORMALIZE_大於600
    - 目的：整理數據，儲存成我要的資料格式
    - 內容：對前10driver大於600秒的旅次，切窗格，進行計算分數、label
    - score.csv > 含 各軸分數、六軸總分
    - X_n.csv > 正規後 儲存
    - X.csv > 正規後、切視窗後reshape(-1,18) 儲存
    - Y_level3_05s > 閾值0.5的風險等級標籤(六軸)
    - Y_level3_1s > 閾值1的風險等級標籤(六軸)
    - Y_level3_15s > 閾值1.5的風險等級標籤(六軸)
    - Y_level3_05s_y > 閾值05的風險等級標籤(單y軸)
    - Y_level3_1s_y > 閾值1的風險等級標籤(單y軸)
    - Y_level3_15s_y > 閾值1.5的風險等級標籤(單y軸)

### Phase 2、3

- Dataset.py
    - 注意：
        - 有修改過，需先檢查確認路徑、內容
- >600_3C05_RNN_物件化_建立lstmgru.ipynb
    - 目的：建立LSTM、GRU、SIMPLERNN、CNN，並且儲存參數做後續使用
    - 寫成物件導向
    - 讀取資料、製作one-hotincoding
    - 模型建立
    - 作圖
- >600_3C05_正式AE_EN_model.ipynb
    - 目的：建立AE、EN-LSTM、EN-GRU
- >600_3C05_正式EN + RF.ipynb
    - 目的：建立EN-RF
- >600_3C05_正式EN +CNN.ipynb
    - 目的：建立EN-CNN
- >600_3C5_DE_model.ipynb
    - 目的：建立DE各個模型

### Phase 4

- 用以旅次為單位的資料
- 取用實驗三訓練好的最佳模型，進行預測
    - RAW model > 選RF
    - RAF model > RAF-RF
    - DE model > DE-RF
- 3C05_RF_predict.ipynb
- 3C05_RAF_RF_predict.ipynb
- 3C05_DE_RF_predict.ipynb
- sample_and_journey_based_cm_RF_05s
    - 先做sample based 檢視與phase3的預測結果有沒有相同
    - 進行journey based 分析
        - 分析risk rule 真實結果 & 預測結果
        - 作圖
- sample_and_journey_based_cm_RAF_RF_05s
- sample_and_journey_based_cm_DE_RF_05s
