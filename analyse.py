#coding=utf-8
import pandas as pd
test_data = pd.read_csv('./data/test.csv')
sales_data = pd.read_csv('./data/sales_train_tmp.csv')
sales_data = sales_data[sales_data['date_block_num'] == 0]
sales_data = sales_data.groupby(['date_block_num','shop_id','item_id'],as_index=False).sum()
sales_data = sales_data.drop(['date_block_num','item_price'],axis = 1)

save_path = './data/submission.csv'
cnt = 0
d = {'ID': [], 'item_cnt_month': []}
for indexs in test_data.index:
    d['ID'].append(test_data.loc[indexs].values[0])
    shop_id = test_data.loc[indexs].values[1]
    item_id = test_data.loc[indexs].values[2]
    result_list = sales_data[(sales_data['shop_id'] == shop_id) & (sales_data['item_id'] == item_id)]['item_cnt_day'].tolist()
    if (len(result_list) == 0):
        d['item_cnt_month'].append(0.5)
        cnt += 1
        continue
    else:
        d['item_cnt_month'].append(result_list[0])
        cnt += 1
result = pd.DataFrame(d, columns=['ID', 'item_cnt_month'])
result.to_csv(save_path,index=0)
# print(sales_data)