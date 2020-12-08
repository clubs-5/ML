#!/usr/bin/env python
# coding: utf-8

# ### 使用Model的方法

from joblib import load
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#讀取模型
model = load('./Model/Logistic_Regression.joblib') 
print(model)



df = pd.read_csv('./Predict_Data/2020_series_tmp.csv')
#print(df)



#載入2020年資料
X = df.drop(['Title','Won'],axis=1)
y = df['Won']

# 特徵縮放
scaler = preprocessing.StandardScaler().fit(X)

#標準化 X
X_nor = scaler.transform(X)

#用以訓練好的模型進行預測
y_pred = model.predict(X_nor)

# 查看準確度
accuracy = accuracy_score(y_pred, y)
print(accuracy)

#查看混淆矩陣，判斷實際預測情況
confusion_matrix = confusion_matrix(y, y_pred)
print(confusion_matrix)

#畫出混淆矩陣
plot_confusion_matrix(model, X_nor, y,
                      cmap=plt.cm.Reds)
#預測結果
print(y_pred)

# ### 得獎影集

# 找出won 的 index, 對應回原本的 dataframe index

result = y_pred.tolist() #numpy array to list

index_list = []

for index,value in enumerate(result):
    if value == 1:
        index_list.append(index)

print("The prediction of  the Primetime Emmy's Award in 2020")
#列出預測的得獎影集
for i in index_list:
    print(df.at[i,'Title'])





