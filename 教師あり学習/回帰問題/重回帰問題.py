#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[8]:


#重回帰モデル
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

# 説明変数は説明変数は「MEDV:価格」以外全て
# 目的変数を「MEDV:価格」としてデータを取り出す
x = boston_df.drop('MEDV', axis = 1)
y = boston_df["MEDV"].values

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)

#単回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train_std, y_train)
# 標準偏回帰係数
#標準偏回帰係数の大きさ（絶対値）順に並び替えて確認
display(pd.DataFrame({"Coefficients":model.coef_,"Abs_Coefficients":np.abs(model.coef_)}).sort_values(by='Abs_Coefficients', ascending=False))
       
#学習データに対するMAE
y_pred = model.predict(x_train_std)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)
#評価データに対するMAE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train_std)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train_std)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)

#実際の価格と予測した価格をプロット
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('実際の価格', fontname="MS Gothic")
plt.ylabel('予測した価格', fontname="MS Gothic")
plt.scatter(y_test,y_pred)
plt.plot([0,50],[0,50],color="red",ls="--")
plt.show()


# In[9]:


#重回帰モデル
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

# 説明変数は「RM:部屋の数, LSTAT:低所得者の割合, DIS:ボストン市の雇用施設からの距離」
# 目的変数を「MEDV:価格」としてデータを取り出す
x = boston_df[['RM', 'LSTAT']].values 
y = boston_df["MEDV"].values

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)

#単回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train_std, y_train)
# 標準偏回帰係数
#標準偏回帰係数の大きさ（絶対値）順に並び替えて確認
display(pd.DataFrame({"Coefficients":model.coef_,"Abs_Coefficients":np.abs(model.coef_)}).sort_values(by='Abs_Coefficients', ascending=False))
       
#学習データに対するMAE
y_pred = model.predict(x_train_std)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)
#評価データに対するMAE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train_std)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train_std)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)

#実際の価格と予測した価格をプロット
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('実際の価格', fontname="MS Gothic")
plt.ylabel('予測した価格', fontname="MS Gothic")
plt.scatter(y_test,y_pred)
plt.plot([0,50],[0,50],color="red",ls="--")
plt.show()


# In[11]:


#重回帰モデル
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

# 説明変数は「RM:部屋の数, LSTAT:低所得者の割合, DIS:ボストン市の雇用施設からの距離」
# 目的変数を「MEDV:価格」としてデータを取り出す
x = boston_df[['RM', 'LSTAT', 'DIS']].values 
y = boston_df["MEDV"].values

#データセットを学習データと評価データに分ける
#全体の30%をテストデータにする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)

#単回帰モデル
model = LinearRegression()
# モデルの学習
model.fit(x_train_std, y_train)
# 標準偏回帰係数
#標準偏回帰係数の大きさ（絶対値）順に並び替えて確認
display(pd.DataFrame({"Coefficients":model.coef_,"Abs_Coefficients":np.abs(model.coef_)}).sort_values(by='Abs_Coefficients', ascending=False))
       
#学習データに対するMAE
y_pred = model.predict(x_train_std)
mae    = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)
#評価データに対するMAE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

#学習データに対するMSE
y_pred = model.predict(x_train_std)
mae    = mean_squared_error(y_train, y_pred)
print("MSE for train data=",mae)
#評価データに対するMSE
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
mae        = mean_squared_error(y_test, y_pred)
print("MSE for test data=",mae)

#学習データに対するR
y_pred = model.predict(x_train_std)
R      = r2_score(y_train, y_pred)
print("R for train data=", R)
#評価データに対するR
x_test_std = sc.transform(x_test) #評価データの標準化
y_pred     = model.predict(x_test_std)
R          = r2_score(y_test, y_pred)
print("R for test data=", R)

#実際の価格と予測した価格をプロット
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('実際の価格', fontname="MS Gothic")
plt.ylabel('予測した価格', fontname="MS Gothic")
plt.scatter(y_test,y_pred)
plt.plot([0,50],[0,50],color="red",ls="--")
plt.show()


# In[ ]:




