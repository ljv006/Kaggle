#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

test_data = pd.read_csv('./data/test.csv')
sales_data = pd.read_csv('./data/sales_train.csv')
# sales_data = sales_data[sales_data['date_block_num'] < 3]
sales_data = sales_data.groupby(['date_block_num','shop_id','item_id'],as_index=False).sum()
sales_data = sales_data.drop(['date_block_num','item_price'],axis = 1)
x = sales_data.iloc[:, 0:2].values
y = sales_data.iloc[:, 2:3].values
# print(sales_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0, random_state = 0)
# model = LinearRegression()

from sklearn import ensemble
print('-'*25 + 'start training' + '-' * 25)
model = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
model.fit(x_train, y_train)
print('-'*25 + 'finish training' + '-' * 25)
# y_pred = model.predict(x_test)
# # 用scikit-learn计算MSE
# print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# # 用scikit-learn计算RMSE
# print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('-' * 25 + 'start reading data' + '-' * 25)
save_path = './data/submission.csv'
cnt = 0
d = {'ID': [], 'item_cnt_month': []}
x_final_test = []
for indexs in test_data.index:
    d['ID'].append(test_data.loc[indexs].values[0])
    shop_id = test_data.loc[indexs].values[1]
    item_id = test_data.loc[indexs].values[2]
    x_final_test.append([shop_id, item_id])
    # result_list = sales_data[(sales_data['shop_id'] == shop_id) & (sales_data['item_id'] == item_id)]['item_cnt_day'].tolist()
    # if (len(result_list) == 0):
    #     d['item_cnt_month'].append(0.5)
    #     cnt += 1
    #     continue
    # else:
    #     d['item_cnt_month'].append(result_list[0])
    #     cnt += 1
print('-' * 25 + 'end reading data' + '-' * 25)
print('-' * 25 + 'start predicting' + '-' * 25)
y_pred = model.predict(x_final_test)
print('-' * 25 + 'end predicting' + '-' * 25)
d['item_cnt_month'] = y_pred.round(0)
result = pd.DataFrame(d, columns=['ID', 'item_cnt_month'])
result.to_csv(save_path,index=0)
# print(sales_data)