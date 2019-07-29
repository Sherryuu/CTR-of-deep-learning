import json 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = json.load(open('../data/data_listby_user_0.json'))
new_data = []
for x in data:
	for y in x:
		new_data.append(y)
new_data = np.array(new_data)
print(new_data.shape)
train,test = train_test_split(new_data,test_size=0.2)
train_data = pd.DataFrame(train,columns=['col_' + str(i) for i in range(24)])
train_data.to_csv('../data/train.csv')
test_data = pd.DataFrame(test,columns=['col_' + str(i) for i in range(24)])
test_data.to_csv('../data/test.csv')

# res = [[] for i in range(24)]
# len_all = 0
# for x in data:
# 	# print (x)
# 	for y in x:
# 		len_all += 1
# 		for j in range(len(y)):
# 			res[j].append(y[j])
# # 		res[j].append(x[j])
# len_res = []
# for y in res:
#  	len_res.append(len(set(y)))
# print (len_res)
# print (len_all)
# print (data[2])