#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# scikit-learnについてくるサンプルデータ
# ボストン住宅価格データセット
# データセットの読み込み
boston = load_boston()

# boston.data: 説明変数
# boston.target: 目的変数
# boston.feature_names: 説明変数名
# 説明変数(boston.data)をDataFrameに保存
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names )
boston_df["MEDV"] = boston.target

#データの説明
print(boston.DESCR)               
display(boston_df)


# In[16]:


# 線形回帰モデル
from sklearn.linear_model import LinearRegression
# ホールドアウト法
from sklearn.model_selection import train_test_split
# 平均絶対誤差
from sklearn.metrics import mean_absolute_error
# 平均二乗誤差
from sklearn.metrics import mean_squared_error
# 決定係数
from sklearn.metrics import r2_score
# 標準化
from sklearn.preprocessing import StandardScaler
# グラフ描画
import matplotlib.pyplot as plt

# 説明変数を「RM:部屋の数」, 目的変数を「MEDV:価格」としてデータを取り出す
# 説明変数（Numpyの配列）
x = boston_df[['RM']].values
# 目的変数（Numpyの配列）
y = boston_df['MEDV'].values   

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#線形回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train, y_train)
# 説明変数の係数を出力
print('coefficient', model.coef_[0]) 
# 切片を出力
print('intercept', model.intercept_) 

#データの上に、求まった単回帰モデルの直線を描画
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('部屋の数', fontname="MS Gothic")
plt.ylabel('価格', fontname="MS Gothic")
plt.scatter(x_train,y_train,label = "Train")
plt.scatter(x_test,y_test,label   = "Test")
plt.plot(x,model.predict(x),color = "red")
plt.legend()
plt.show()

#学習データに対するMAE
y_pred = model.predict(x_train)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)

#評価データに対するMAE
y_pred = model.predict(x_test)
mae    = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
y_pred     = model.predict(x_test)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
y_pred     = model.predict(x_test)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)                      


# In[17]:


# 線形回帰モデル
from sklearn.linear_model import LinearRegression
# ホールドアウト法
from sklearn.model_selection import train_test_split
# 平均絶対誤差
from sklearn.metrics import mean_absolute_error
# 平均二乗誤差
from sklearn.metrics import mean_squared_error
# 決定係数
from sklearn.metrics import r2_score
# 標準化
from sklearn.preprocessing import StandardScaler
# グラフ描画
import matplotlib.pyplot as plt

# 説明変数を「LSTAT:低所得者の割合」, 目的変数を「MEDV:価格」としてデータを取り出す
# 説明変数（Numpyの配列）
x = boston_df[['LSTAT']].values
# 目的変数（Numpyの配列）
y = boston_df['MEDV'].values 

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#線形回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train, y_train)
# 説明変数の係数を出力
print('coefficient', model.coef_[0]) 
# 切片を出力
print('intercept', model.intercept_) 

#データの上に、求まった単回帰モデルの直線を描画
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('低所得者の割合', fontname="MS Gothic")
plt.ylabel('価格', fontname="MS Gothic")
plt.scatter(x_train,y_train,label = "Train")
plt.scatter(x_test,y_test,label   = "Test")
plt.plot(x,model.predict(x),color = "red")
plt.legend()
plt.show()

#学習データに対するMAE
y_pred = model.predict(x_train)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)

#評価データに対するMAE
y_pred = model.predict(x_test)
mae    = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
y_pred     = model.predict(x_test)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
y_pred     = model.predict(x_test)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)


# In[18]:


# 線形回帰モデル
from sklearn.linear_model import LinearRegression
# ホールドアウト法
from sklearn.model_selection import train_test_split
# 平均絶対誤差
from sklearn.metrics import mean_absolute_error
# 平均二乗誤差
from sklearn.metrics import mean_squared_error
# 決定係数
from sklearn.metrics import r2_score
# 標準化
from sklearn.preprocessing import StandardScaler
# グラフ描画
import matplotlib.pyplot as plt

# 説明変数を「DIS:ボストン市の雇用施設からの距離」, 目的変数を「MEDV:価格」としてデータを取り出す
# 説明変数（Numpyの配列）
x = boston_df[['DIS']].values
# 目的変数（Numpyの配列）
y = boston_df['MEDV'].values   

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#線形回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train, y_train)
# 説明変数の係数を出力
print('coefficient', model.coef_[0]) 
# 切片を出力
print('intercept', model.intercept_) 

#データの上に、求まった単回帰モデルの直線を描画
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('ボストン市の雇用施設からの距離', fontname="MS Gothic")
plt.ylabel('価格', fontname="MS Gothic")
plt.scatter(x_train,y_train,label = "Train")
plt.scatter(x_test,y_test,label   = "Test")
plt.plot(x,model.predict(x),color = "red")
plt.legend()
plt.show()

#学習データに対するMAE
y_pred = model.predict(x_train)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)

#評価データに対するMAE
y_pred = model.predict(x_test)
mae    = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
y_pred     = model.predict(x_test)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
y_pred     = model.predict(x_test)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)                  


# In[ ]:




