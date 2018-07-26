import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import pandas.util.testing as tm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


#pd.set_option('mpl_style','default')

#数据从数据库读取
pymysql.install_as_MySQLdb()
engine = create_engine(
    'mysql+mysqldb://root:root@192.168.10.14:3306/com66nao_cloud?charset=utf8')
df1 = pd.read_sql('select user_id as user_truename,train_score as score ,train_time,cogn_task.id as gamename ,cloud_cat.name as firstbrain from user_train_history join cogn_task on cogn_task.id=user_train_history.game_id join cloud_cat on cloud_cat.id=cogn_task.label and cloud_cat.id', engine)
df2 = df1[['user_truename','gamename', 'score','train_time']]  # 只选取有实际作用的列
#因为在线数据库中有的游戏其实应该是没有上线，一切数值均为0，不应该被纳入考虑，因而删除空值
df3 = df2 [df2['score']>0]
df4 = df3[ df3['train_time']>15] #删除哪些游戏时间小于15s的
df4 = df4.dropna()
df4['mean_score'] = df4['score']/df4['train_time']
data = df4[['user_truename','mean_score','gamename','score','train_time']]
tables1 = pd.pivot_table(data, index='gamename', values='mean_score', aggfunc=lambda x: np.percentile(x,25)-1.5*(np.percentile(x,75)-np.percentile(x,25)))["mean_score"]
tables2 = pd.pivot_table(data, index='gamename', values='mean_score', aggfunc=lambda x: np.percentile(x,75)+1.5*(np.percentile(x,75)-np.percentile(x,25)))["mean_score"]
pd_tables = pd.DataFrame([tables1,tables2])
pd_tables = pd_tables.T
pd_tables.columns = ['lowerlimit','upperlimit']
pd_tables["gamename"] = pd_tables.index
score_table = pd.merge(pd_tables,data, on="gamename")

score_table1 = score_table[score_table['upperlimit']>score_table['mean_score']]
score_table2 = score_table1[score_table1['mean_score']>score_table1['lowerlimit']]


listgamename = list(set(list(data['gamename'])))
for i in listgamename:
    singledata = data[data['gamename']== i]['mean_score']
    singledata = singledata.reset_index()['mean_score']
    df_singledata = singledata.to_frame(name=None)
    #plt.figure(1)
    plt.figure
    p = df_singledata.boxplot()
    plt.savefig("E:/box_plot/%s.png"%(i))
    plt.show()

grouped = score_table2['mean_score'].groupby(score_table2['gamename'])
des_after = grouped.describe()
