# -*- coding: utf-8 -*-
#using python 3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import  Embedding,Dropout,Dense,Reshape,Merge,Concatenate
from sqlalchemy import create_engine
import pymysql



k=128
#数据预处理，将原始数据划分为6个等级
pymysql.install_as_MySQLdb()
engine = create_engine(
    'mysql+mysqldb://root:root@192.168.10.14:3306/com66nao_cloud?charset=utf8')
df4 = pd.read_sql('select user_id as user_truename,train_score as score ,cogn_task.name as name ,cloud_cat.name as firstbrain from user_train_history join cogn_task on cogn_task.id=user_train_history.game_id join cloud_cat on cloud_cat.id=cogn_task.label', engine)
df1 = df4[['user_truename', 'name', 'score']]  # 只选取有实际作用的列
#因为在线数据库中有的游戏其实应该是没有上线，一切数值均为0，不应该被纳入考虑，因而删除空值
df5 = df1 [df1>0]
df6 = df5.dropna()
grouped = df6['score'].groupby([df6['user_truename'], df6['name']])
df2 = grouped.median()
df3 = df2.reset_index()


old_set=np.unique(df3['user_truename'])
old_list=list(old_set)
new_id=np.arange(len(old_list))
df3['new_id']=df3['user_truename'].replace(old_list,new_id)
df3['new_id']=df3['new_id']+1

print('OK')

old_gameset = np.unique(df3['name'])
old_gamelist = list(old_gameset)
fid = np.arange(len(old_gamelist))
df3['fid'] = df3['name'].replace(old_gamelist,fid)
df3['fid'] = df3['fid']+1


print("ok")
print('ik')


tables1 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.90)["score"]
tables2 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.70)["score"]
tables3 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.50)["score"]
tables4 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.30)["score"]
tables5 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.10)["score"]
total_score = pd.DataFrame([tables1, tables2, tables3, tables4, tables5])
final_score = total_score.T

final_score.columns = ['a', 'b', 'c', 'd', 'e']

final_score["fid"] = final_score.index

score_table = pd.merge(final_score, df3, on="fid")
print("ok")
score_table.loc[(score_table["score"] < score_table['e']), 'score'] = 6  #
score_table.loc[(score_table["score"] >= score_table['e']) & (score_table["score"] < score_table['d']), 'score'] = 5
score_table.loc[(score_table["score"] >= score_table['d']) & (score_table["score"] < score_table['c']), 'score'] = 4
score_table.loc[(score_table["score"] >= score_table['c']) & (score_table["score"] < score_table['b']), 'score'] = 3
score_table.loc[(score_table["score"] >= score_table['b']) & (score_table["score"] < score_table['a']), 'score'] = 2
score_table.loc[(score_table["score"] >= score_table['a']), 'score'] = 1  # 将所有的得分离散化在等级里面
score_table = score_table[['new_id', 'fid', 'score']]  # 只选取有实际作用的列
#n数据读取结束



score_table.to_csv('E:/lf/score_table.csv',index = None)

n_users=np.max(score_table['new_id'])
n_movices=int(np.max(score_table['fid']))
print([n_users,n_movices,len(score_table)])
score_table.to_csv('E:/scoretable.csv',index = None)
model1=Sequential()
model1.add(Embedding(n_users+1,k,input_length=1))
model1.add(Reshape((k,)))
model2=Sequential()
model2.add(Embedding(n_movices+1,k,input_length=1))
model2.add(Reshape((k,)))
model=Sequential()
model.add(Merge([model1,model2],mode='concat'))#然后加入Dropout 和relu 这个非线性变换项，构造多层深度模型。
#model.add(Concatenate([model1, model2]))
#x = concatenate([a, b], axis=-1)
model.add(Dropout(0.5))
model.add(Dense(k, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k / 4), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k / 16), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation = 'softmax'))#因为是预测连续变量评分，最后一层直接上线性变化
model.compile(loss='categorical_crossentropy',optimizer='adam')




users=score_table['new_id'].values
movices=score_table['fid'].values
y=score_table['score'].values

from keras.utils import np_utils
nb_classes = 6
y_train = np_utils.to_categorical(y, nb_classes)

from sklearn.model_selection import train_test_split
x_train_users, x_test_users,x_train_movices, x_test_movices,y_train, y_test = train_test_split(users, movices, y,test_size=0.2, random_state=None)
x_train = [x_train_users,x_train_movices]
x_test = [x_test_users,x_test_movices]
x = [users,movices]
model.fit(x_train,y_train,batch_size=100,epochs=10)


print("training is start")
model.save_weights('model629.h5')
json_string = model.to_json()
with open("model629.json", "w") as json_file:
    json_file.write(json_string)




traincost = model.evaluate(x_train,y_train)
print('traincost:',traincost)

testcost = model.evaluate(x_test,y_test)
print('testcost:',testcost)

allcost = model.evaluate(x,y)
print('allcost:',allcost)


#这是总的均方误差，而不是测试集的均方误差
sum =0
predictions = []
for i in range(score_table.shape[0]):
    predictions.append(model.predict([np.array([score_table['new_id'][i]]), np.array([score_table['fid'][i]])]))
    sum += (score_table['score'][i] - model.predict([np.array([score_table['new_id'][i]]), np.array([score_table['fid'][i]])])) ** 2
mse = math.sqrt(sum/score_table.shape[0])
print("手算均方误差是",mse)


pred1 = model.predict([np.array([10]),np.array([1])])
print('[11,1]的预测值：',[np.array([10]),np.array([1])],'  ',pred1)

pred2 = model.predict([np.array([score_table['new_id'][0]]), np.array([score_table['fid'][0]])])
print('[11,1]的预测值：',[np.array([10]),np.array([1])],'  ',pred2)

pred3 = model.predict([np.array([users[10]]),np.array([movices[1]])])
print('[11,1]的预测值：',[np.array([10]),np.array([1])],'  ',pred3)

pd_predictions = pd.Series(predictions)
pd_predictions.to_csv('E:/lf/pd_predictions.csv',index = None)

print('ok')

