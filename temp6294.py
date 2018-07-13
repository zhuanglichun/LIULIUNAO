# -*- coding: utf-8 -*-
#using python 3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import re
from sqlalchemy import create_engine
import re
import pymysql
pymysql.install_as_MySQLdb()

import numpy as np
import pandas as pd
score_table = pd.read_csv('E:/scoretable.csv')

n_users=np.max(score_table['new_id'])
n_movices=int(np.max(score_table['fid']))
indexmax = int(score_table.shape[0])
score_table_len = len(score_table)
scorearray = np.zeros((n_users,n_movices))

print('ok')
print('ok')
for index in range(0,indexmax):
    row =  int(score_table.loc[index]['new_id']-1)
    column = int( score_table.loc[index]['fid']-1)
    scorearray[row][column] =  score_table.loc[index]['score']
df_s = pd.DataFrame(scorearray)
#df_s.to_csv('E:/scorearrayspyder2.csv',index)
print('ok')
print('ok')
l = [ (indexs,i) for indexs in df_s.index for i in
     range(len(df_s.loc[indexs].values)) if(df_s.loc[indexs].values[i] ==0)]
df_location = pd.DataFrame(l)+1
df_location.to_csv('E:/nanlocation.csv',index =None)
print('ok')
print('ok')

from keras.models import model_from_json
json_file = open('model629.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model629.h5")
pred_array = np.zeros((n_users,n_movices))

# for s in range(0,len(df_location)):
# #     i = df_location.loc[s][0]
# #     j = df_location.loc[s][1]
# #     a = i-1
# #     b = j-1
# #     pred_array[a][b] = model.predict([np.array([i]),np.array([j])])
# #
# # df_pred = pd.DataFrame(pred_array)
# # df_pred.to_csv('E:/pred.csv',index = None)
# #
# # print('ok')
# # print('ok')
# # df_preddata = df_s+df_pred
# # df_preddata.to_csv('E:/preddata.csv',index = None)
# # print('ok')

users=score_table['new_id'].values
movices=score_table['fid'].values
y=score_table['score'].values
pred1 = model.predict([np.array([11]),np.array([1])])
pred2 = model.predict([np.array([users[11]]), np.array([movices[1]])])

pred3 = model.predict([np.array([40]),np.array([1])])
pred4 = model.predict([np.array([users[40]]), np.array([movices[1]])])


print('pred1:',pred1)
print('pred2:',pred2)
print('pred3:',pred3)
print('pred4:',pred4)