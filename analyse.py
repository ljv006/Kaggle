#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

test_data = pd.read_csv('./data/test.csv')
sales_data = pd.read_csv('./data/sales_train.csv')
sales_data['item_cnt_day'] = sales_data['item_cnt_day'].clip(0,20)
sales_data = sales_data.groupby(['date_block_num','shop_id','item_id'],as_index=False).sum()
sales_data['item_cnt_day'] = sales_data['item_cnt_day'].clip(0,20)
sales_data = sales_data.drop(['date_block_num','item_price'],axis = 1)
x = sales_data.iloc[:, 0:2].values
y = sales_data.iloc[:, 2:3].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0, random_state = 0)
# model = LinearRegression()

from sklearn import ensemble
print('-' * 33 + 'start training' + '-' * 33)
# from sklearn import svm
# model = svm.SVR()
model = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
model.fit(x_train, y_train)
print('-'*32 + 'finish training' + '-' * 32)
# y_pred = model.predict(x_test)
# # 用scikit-learn计算MSE
# print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# # 用scikit-learn计算RMSE
# print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('-' * 31 + 'start reading data' + '-' * 31)
save_path = './data/submission.csv'

d = {'ID': [], 'item_cnt_month': []}
x_final_test = []
for indexs in test_data.index:
    d['ID'].append(test_data.loc[indexs].values[0])
    shop_id = test_data.loc[indexs].values[1]
    item_id = test_data.loc[indexs].values[2]
    x_final_test.append([shop_id, item_id])
print('-' * 32 + 'end reading data' + '-' * 32)
print('-' * 33 + 'start predicting' + '-' * 33)
y_pred = model.predict(x_final_test)
print('-' * 33 + 'end predicting' + '-' * 33)
d['item_cnt_month'] = y_pred / 20.0
result = pd.DataFrame(d, columns=['ID', 'item_cnt_month'])
result.to_csv(save_path,index=0)
# print(sales_data)