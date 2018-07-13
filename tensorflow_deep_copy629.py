# -*- coding: utf-8 -*-
#using python 3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import  Embedding,Dropout,Dense,Reshape,Merge
from sqlalchemy import create_engine
import pymysql

k=128
pymysql.install_as_MySQLdb()
# 创建mysql连接引擎
# engine = create_engine('mysql+mysqldb://username:password@host:port/dbname?charset=utf8')
#查询数据并转为pandas.DataFrame，指定DataFrame的index为数据库中的id字段
engine = create_engine(
    'mysql+mysqldb://root:root@192.168.10.14:3306/com66nao_cloud?charset=utf8')
# df = pd.read_sql('SELECT * FROM game1', engine, index_col='id')
#将修改后的数据追加至原表,index=False代表不插入索引，因为数据库中id字段为自增字段
df4 = pd.read_sql('select user_id as user_truename,train_score as score ,cogn_task.name as name ,cloud_cat.name as firstbrain from user_train_history join cogn_task on cogn_task.id=user_train_history.game_id join cloud_cat on cloud_cat.id=cogn_task.label', engine)



df1 = df4[['user_truename', 'name', 'score']]  # 只选取有实际作用的列
grouped = df1['score'].groupby([df1['user_truename'], df1['name']])
# print(grouped)
df2 = grouped.median()
# print(df2)
df3 = df2.reset_index()

print('ok')
old_set=np.unique(df3['user_truename'])
old_list=list(old_set)
new_id=np.arange(len(old_list))
df3['new_id']=df3['user_truename'].replace(old_list,new_id)
df3['new_id']=df3['new_id']+1
df3
print('OK')

old_gameset = np.unique(df3['name'])
old_gamelist = list(old_gameset)
fid = np.arange(len(old_gamelist))
df3['fid'] = df3['name'].replace(old_gamelist,fid)
df3['fid'] = df3['fid']+1

df3
print("ok")
print('ik')


# tables1=pd.pivot_table(df3,index='name',values='score',aggfunc=[lambda x:(np.max(x)-np.min(x))*0.9,lambda x:(np.max(x)-np.min(x))*0.8,lambda x:(np.max(x)-np.min(x))*0.7,lambda x:(np.max(x)-np.min(x))*0.6,lambda x:(np.max(x)-np.min(x))*0.5,np.mean])
# tables1=pd.pivot_table(df3,index='name',values='score',aggfunc=pd.qcut(x,3))
tables1 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.80)["score"]
tables2 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.70)["score"]
tables3 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.50)["score"]
tables4 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.30)["score"]
tables5 = pd.pivot_table(df3, index='fid', values='score', aggfunc=lambda x: (np.max(x) - np.min(x)) * 0.10)["score"]
total_score = pd.DataFrame([tables1, tables2, tables3, tables4, tables5])
final_score = total_score.T
# print("u word is ")
final_score.columns = ['a', 'b', 'c', 'd', 'e']
# print(final_score.columns)
# final_score.drop([0],axis=0)
final_score["fid"] = final_score.index
# print(final_score)
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
n_users=np.max(score_table['new_id'])
n_movices=int(np.max(score_table['fid']))
print("all right")
print('ok')
print([n_users,n_movices,len(score_table)])


print('ok')
score_table.to_csv('E:/scoretable.csv',index = None)

print('OK')
print('ok')

# plt.hist(score_table['fid'])
# plt.show()
# print(np.mean(score_table['fid']))
model1=Sequential()
model1.add(Embedding(n_users+1,k,input_length=1))
model1.add(Reshape((k,)))
model2=Sequential()
model2.add(Embedding(n_movices+1,k,input_length=1))
model2.add(Reshape((k,)))
model=Sequential()
model.add(Merge([model1,model2],mode='concat'))#然后加入Dropout 和relu 这个非线性变换项，构造多层深度模型。
#model=Concatenate()([model1.output, model2.output])

model.add(Dropout(0.2))
model.add(Dense(k, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k / 4), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k / 16), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'linear'))#因为是预测连续变量评分，最后一层直接上线性变化
model.compile(loss='mse',optimizer='adam')



from sklearn.utils import shuffle
#score_table = shuffle(score_table)
users=score_table['new_id'].values
movices=score_table['fid'].values
y=score_table['score'].values
# X_train=[users,movices]
# y_train=score_table['score'].values
# x=[users,movices]
p = 0.8 #设置训练数据比例

a =int( len (score_table["new_id"]))
alpha = int(a*p)
#print(a)
#print(score_table["new_id"])


users_train=users[:alpha]
users_test=users[alpha:]
movices_train=movices[:alpha]
movices_test=movices[alpha:]
x_train=[users_train,movices_train]
x_test=[users_test,movices_test]
y_train=y[:alpha]
y_test=y[alpha:]
print('ok')

model.fit(x_train,y_train,batch_size=100,epochs=100)
# model.fit(x,y,batch_size=100,epochs=1)
#i=906          #现在的表示的是第十个用户对第4个电影的得分。
#j=81
#list_users = list(users)
i=11          #现在的表示的是第十一个用户对第1个电影的得分。
j=1



pred = model.predict([np.array([i]),np.array([j])])
# print("训练开始，首先将原始数据集随机打乱，取到前80%是训练集，后%20是测试集合")
print("training is start")
print("预测的是")
model.save_weights('model629.h5')
# model.load_weights('test.h5',by_name=True)
json_string = model.to_json()
with open("model629.json", "w") as json_file:
    json_file.write(json_string)
# model = model_from_json(json_string)
# model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it 使用保存的话，必须使用h5py
#注意在嵌套的网络中keras 是要参数和结构分开保存的
# del model  # deletes the existing model
# load
# model = load_model('my_model.h5')
print(pred)
sum =0
for i in range(score_table.shape[0]):
    sum += (score_table['score'][i] - model.predict([np.array([score_table['new_id'][i]]), np.array([score_table['fid'][i]])])) ** 2
mse = math.sqrt(sum/score_table.shape[0])
print("下面的是均方误差是")

#cost2 = model.evaluate(x_train,y_train)
#print('train_cost:',cost2)
print(mse)
print('\nTesting ------------')
print("下面是测试集合的loss")
cost = model.evaluate(x_test,y_test, batch_size=100)
print('test cost:', cost)
print('ok')
print('ok')

#pred1 = model.predict([np.array([users[11]]), np.array([movices[1]])])
#print(pred1)

#print('pred1:',pred1)
pred = model.predict([np.array([11]),np.array([1])])
print(pred)

pred1 = model.predict([np.array([users[11]]), np.array([movices[1]])])
print('pred1:',pred1)

pred2 = model.predict([np.array([score_table['new_id'][0]]), np.array([score_table['fid'][0]])])
print('pred2:',pred2)


print('ok')
print(users[11])
print('ok')