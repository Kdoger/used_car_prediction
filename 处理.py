#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['KaiTi']  #中文
plt.rcParams['axes.unicode_minus'] = False   #负号


# ## 读取数据

# In[79]:


columns=['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color','cityid', 'carCode', 'transferCount','seatings',
         'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear', 'displacement','gearbox','oiltype', 'newprice', 
         'AF1', 'AF2', 'AF3', 'AF4', 'AF5','AF6', 'AF7', 'AF8', 'AF9', 'AF10', 'AF11','AF12', 'AF13', 'AF14','AF15', 'price']

data=pd.read_csv('附件1：估价训练数据.txt',sep='\t',header=None,names=columns) 
data2=pd.read_csv('附件2：估价验证数据.txt',sep='\t',header=None,names=columns[:-1]) 
data.head()


# In[80]:


data.infer_objects()
data2.infer_objects()
data.info() ,data2.info()#查看数据基础信息


# ## 数据清洗

# In[81]:


#观察缺失值
import missingno as msno
msno.matrix(data)


# In[82]:


msno.matrix(data2)


# In[83]:


#删除id序号
data.drop('carid',axis=1,inplace=True)
ID=data2['carid']


# In[84]:


#若是有一行全为空值就删除
data.dropna(how='all',inplace=True)
data2.dropna(how='all',inplace=True)


# In[85]:


#取值唯一的变量删除
for col in data.columns:
    if len(data[col].value_counts())==1:
        print(col)
        data.drop(col,axis=1,inplace=True)


# In[86]:


#缺失到一定比例就删除
miss_ratio=0.15
for col in data.columns:
    if  data[col].isnull().sum()>data.shape[0]*miss_ratio:
        print(col)
        data.drop(col,axis=1,inplace=True)


# In[87]:


#查看数值型数据,
pd.set_option('display.max_columns', 30)
data.select_dtypes(exclude=['object']).head()


# #有的是分类数据，有的是日期,需要处理一下

# In[88]:


#删除不要数值变量,不知道含义
data.drop('AF13',axis=1,inplace=True)


# In[89]:


#计算异众比例
variation_ratio_s=0.1
for col in data.select_dtypes(exclude=['object']).columns:
    df_count=data[col].value_counts()
    kind=df_count.index[0]
    variation_ratio=1-(df_count.iloc[0]/len(data[col]))
    if variation_ratio<variation_ratio_s:
        print(f'{col} 最多的取值为{kind}，异众比例为{round(variation_ratio,4)},太小了，没区分度删掉')
        data.drop(col,axis=1,inplace=True)


# In[90]:


#查看非数值型数据
data.select_dtypes(exclude=['int64','float64']).head()


# #有用的等下构建特征去做特征，该独热独热，没用的才删除

# In[91]:


#删除不要的字符变量
data.drop(['licenseDate','AF12'],axis=1,inplace=True)


# In[92]:


#总结一下现在的变量，并给测试集也筛一下
columns=data.columns
print(columns)
data2=data2[columns[:-1]]   #y值不要


# ### 填充缺失值

# In[93]:


#均值填充，中位数填充，众数填充
#前填充，后填充
data.fillna(data.median(),inplace=True)   #mode,mean
data.fillna(method='ffill',inplace=True)   #pad,bfill/backfill

data2.fillna(data2.median(),inplace=True)   
data2.fillna(method='ffill',inplace=True)


# ## 特征工程

# In[94]:


#首先对训练集取出y
y=data['price']
data.drop(['price'],axis=1,inplace=True)


# In[95]:


#对前面的要处理的变量单独处理，
#时间类型变量的处理，得出二手车年龄，天数
data['age']=(pd.to_datetime(data['tradeTime'])-pd.to_datetime(data['registerDate'])).map(lambda x:x.days)
#测试集也一样变化
data2['age']=(pd.to_datetime(data2['tradeTime'])-pd.to_datetime(data2['registerDate'])).map(lambda x:x.days)


# In[96]:


#然后删除原来的不需要的特征变量
data.drop(['tradeTime','registerDate'],axis=1,inplace=True)
data2.drop(['tradeTime','registerDate'],axis=1,inplace=True)


# In[97]:


#可以映射
# d1={'male':0,'female':1}
# d2={'S':1,'C':2,'Q':3}

# data['Sex']=data['Sex'].map(d1)
# data['Embarked']=data['Embarked'].map(d2)
# data2['Sex']=data2['Sex'].map(d1)
# data2['Embarked']=data2['Embarked'].map(d2)


# In[98]:


#剩下的变量独热处理
data=pd.get_dummies(data)
data2=pd.get_dummies(data2)


# In[99]:


#独热多出来的特征处理
for col in data.columns:
    if col not in data2.columns:
        data2[col]=0


# In[100]:


print(data.shape,data2.shape,y.shape)


# In[101]:


#查看处理完的数据信息
data.info(),data2.info()


# ### 数据画图探索

# In[102]:


#查看特征变量的箱线图分布
columns = data.columns.tolist() # 列表头
dis_cols = 7                   #一行几个
dis_rows = len(columns)
plt.figure(figsize=(4 * dis_cols, 4 * dis_rows))

for i in range(len(columns)):
    plt.subplot(dis_rows,dis_cols,i+1)
    sns.boxplot(data=data[columns[i]], orient="v",width=0.5)
    plt.xlabel(columns[i],fontsize = 20)
plt.tight_layout()
#plt.savefig('特征变量箱线图',formate='png',dpi=500)
plt.show()


# In[103]:


#画密度图，训练集和测试集对比
dis_cols = 5                   #一行几个
dis_rows = len(columns)
plt.figure(figsize=(4 * dis_cols, 4 * dis_rows))

for i in range(len(columns)):
    ax = plt.subplot(dis_rows, dis_cols, i+1)
    ax = sns.kdeplot(data[columns[i]], color="Red" ,shade=True)
    ax = sns.kdeplot(data2[columns[i]], color="Blue",warn_singular=False,shade=True)
    ax.set_xlabel(columns[i],fontsize = 20)
    ax.set_ylabel("Frequency",fontsize = 18)
    ax = ax.legend(["train", "test"])
plt.tight_layout()
#plt.savefig('训练测试特征变量核密度图',formate='png',dpi=500)
plt.show()


# In[104]:


#选出分布不一样的特征,删除
drop_col=['country','AF11_1,3+2','AF11_5']
data.drop(drop_col,axis=1,inplace=True)
data2.drop(drop_col,axis=1,inplace=True)


# In[105]:


# 查看y的分布
#回归问题
plt.figure(figsize=(6,2),dpi=128)
plt.subplot(1,3,1)
y.plot.box(title='响应变量箱线图')
plt.subplot(1,3,2)
y.plot.hist(title='响应变量直方图')
plt.subplot(1,3,3)
y.plot.kde(title='响应变量核密度图')
#sns.kdeplot(y, color='Red', shade=True)
#plt.savefig('响应变量.png')
plt.tight_layout()
plt.show()

# 分类问题
# plt.figure(figsize=(6,2),dpi=128)
# plt.subplot(1,3,1)
# y.value_counts().plot.bar(title='响应变量柱状图图')
# plt.subplot(1,3,2)
# y.value_counts().plot.pie(title='响应变量饼图')
# plt.subplot(1,3,3)
# y.plot.kde(title='响应变量核密度图')
# plt.savefig('响应变量.png')
plt.tight_layout()
# plt.show()


# #上图的y存在着极端值，要筛掉

# ### 异常值处理

# In[106]:


#处理y的异常值
y=y[y <= 200]
plt.figure(figsize=(6,2),dpi=128)
plt.subplot(1,3,1)
y.plot.box(title='响应变量箱线图')
plt.subplot(1,3,2)
y.plot.hist(title='响应变量直方图')
plt.subplot(1,3,3)
y.plot.kde(title='响应变量核密度图')
#sns.kdeplot(y, color='Red', shade=True)
#plt.savefig('响应变量.png')
plt.tight_layout()
plt.show()


# In[107]:


#筛选给x
data=data.iloc[y.index,:]
data.shape


# In[108]:


#X异常值处理，先标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(data)
X2_s = scaler.fit_transform(data2)


# In[109]:


plt.figure(figsize=(20,8))
plt.boxplot(x=X_s,labels=data.columns)
#plt.hlines([-10,10],0,len(columns))
plt.show()


# In[110]:


plt.figure(figsize=(20,8))
plt.boxplot(x=X2_s,labels=data2.columns)
#plt.hlines([-10,10],0,len(columns))
plt.show()


# In[111]:


#异常值多的列进行处理
def deal_outline(data,col,n):   #数据，要处理的列名，几倍的方差
    for c in col:
        mean=data[c].mean()
        std=data[c].std()
        return data[(data[c]>mean-n*std)&(data[c]<mean+n*std)]


# In[112]:


data=deal_outline(data,['newprice'],10)
y=y[data.index]
data.shape,y.shape


# ### 查看相关系数矩阵

# In[113]:


corr = plt.subplots(figsize = (18,16),dpi=128)
corr= sns.heatmap(data.assign(Y=y).corr(method='spearman'),annot=True,square=True)


# In[114]:


corr = plt.subplots(figsize = (18,16),dpi=128)
corr= sns.heatmap(data2.corr(method='spearman'),annot=True,square=True)


# ## 开始机器学习！

# In[115]:


#划分训练集和验证集
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(data,y,test_size=0.2,random_state=0)


# In[116]:


#数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X2_s=scaler.transform(data2)
print('训练数据形状：')
print(X_train_s.shape,y_train.shape)
print('验证测试数据形状：')
(X_val_s.shape,y_val.shape,X2_s.shape)


# In[117]:


#采用十种模型，对比验证集精度
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# In[41]:


#线性回归
model1 = LinearRegression()

#弹性网回归
model2 = ElasticNet(alpha=0.05, l1_ratio=0.5)

#K近邻
model3 = KNeighborsRegressor(n_neighbors=10)

#决策树
model4 = DecisionTreeRegressor(random_state=77)

#随机森林
model5= RandomForestRegressor(n_estimators=500,  max_features=int(X_train.shape[1]/3) , random_state=0)

#梯度提升
model6 = GradientBoostingRegressor(n_estimators=500,random_state=123)

#极端梯度提升
model7 =  XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=0)

#轻量梯度提升
model8 = LGBMRegressor(n_estimators=1000,objective='regression', # 默认是二分类
                      random_state=0)

#支持向量机
model9 = SVR(kernel="rbf")

#神经网络
model10 = MLPRegressor(hidden_layer_sizes=(16,8), random_state=77, max_iter=10000)

model_list=[model1,model2,model3,model4,model5,model6,model7,model8,model9,model10]
model_name=['线性回归','惩罚回归','K近邻','决策树','随机森林','梯度提升','极端梯度提升','轻量梯度提升','支持向量机','神经网络']


# In[42]:


for i in range(10):
    model_C=model_list[i]
    name=model_name[i]
    model_C.fit(X_train_s, y_train)
    s=model_C.score(X_val_s, y_val)
    print(name+'方法在验证集的准确率为：'+str(s))


# #一般集成方法都最好，后面交叉验证、网格搜参都只选三个模型，随机森林，极端梯度提升，轻量梯度提升

# ### 交叉验证

# In[42]:


#回归问题交叉验证，使用拟合优度，mae,rmse,mape 作为评价标准
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape=(abs(y_predict -y_test)/ y_test).mean()
    return mae, rmse, mape
def evaluation2(lis):
    array=np.array(lis)
    return array.mean() , array.std()


# In[44]:


def cross_val(model=None,X=None,Y=None,K=5,repeated=1):
    df_mean=pd.DataFrame(columns=['R2','MAE','RMSE','MAPE']) 
    df_std=pd.DataFrame(columns=['R2','MAE','RMSE','MAPE'])
    for n in range(repeated):
        print(f'正在进行第{n+1}次重复K折.....随机数种子为{n}\n')
        kf = KFold(n_splits=K, shuffle=True, random_state=n)
        R2=[]
        MAE=[]
        RMSE=[]
        MAPE=[]
        print(f"    开始本次在{K}折数据上的交叉验证.......\n")
        i=1
        for train_index, test_index in kf.split(X):
            print(f'        正在进行第{i}折的计算')
            X_train=X.values[train_index]
            y_train=y.values[train_index]
            X_test=X.values[test_index]
            y_test=y.values[test_index]
            model.fit(X_train,y_train)
            score=model.score(X_test,y_test)
            R2.append(score)
            pred=model.predict(X_test)
            mae, rmse, mape=evaluation(y_test, pred)
            MAE.append(mae)
            RMSE.append(rmse)
            MAPE.append(mape)
            print(f'        第{i}折的拟合优度为：{round(score,4)}，MAE为{round(mae,4)}，RMSE为{round(rmse,4)}，MAPE为{round(mape,4)}')
            i+=1
        print(f'    ———————————————完成本次的{K}折交叉验证———————————————————\n')
        R2_mean,R2_std=evaluation2(R2)
        MAE_mean,MAE_std=evaluation2(MAE)
        RMSE_mean,RMSE_std=evaluation2(RMSE)
        MAPE_mean,MAPE_std=evaluation2(MAPE)
        print(f'第{n+1}次重复K折，本次{K}折交叉验证的总体拟合优度均值为{R2_mean}，方差为{R2_std}')
        print(f'                               总体MAE均值为{MAE_mean}，方差为{MAE_std}')
        print(f'                               总体RMSE均值为{RMSE_mean}，方差为{RMSE_std}')
        print(f'                               总体MAPE均值为{MAPE_mean}，方差为{MAPE_std}')
        print("\n====================================================================================================================\n")
        df1=pd.DataFrame(dict(zip(['R2','MAE','RMSE','MAPE'],[R2_mean,MAE_mean,RMSE_mean,MAPE_mean])),index=[n])
        df_mean=pd.concat([df_mean,df1])
        df2=pd.DataFrame(dict(zip(['R2','MAE','RMSE','MAPE'],[R2_std,MAE_std,RMSE_std,MAPE_std])),index=[n])
        df_std=pd.concat([df_std,df2])
    return df_mean,df_std


# In[45]:


model = LGBMRegressor(n_estimators=1000,objective='regression',random_state=0)
lgb_crosseval,lgb_crosseval2=cross_val(model=model,X=data,Y=y,K=3,repeated=5)


# In[46]:


lgb_crosseval


# In[47]:


model = XGBRegressor(n_estimators=1000,objective='reg:squarederror',random_state=0)
xgb_crosseval,xgb_crosseval2=cross_val(model=model,X=data,Y=y,K=3,repeated=5)


# In[48]:


xgb_crosseval


# In[49]:


model = RandomForestRegressor(n_estimators=500,  max_features=int(X_train.shape[1]/3) , random_state=0)
rf_crosseval,rf_crosseval2=cross_val(model=model,X=data,Y=y,K=3,repeated=5)


# In[50]:


rf_crosseval


# In[54]:


plt.subplots(1,4,figsize=(16,3))
for i,col in enumerate(lgb_crosseval.columns):
    n=int(str('14')+str(i+1))
    plt.subplot(n)
    plt.plot(lgb_crosseval[col], 'k', label='LGB')
    plt.plot(xgb_crosseval[col], 'b-.', label='XGB')
    plt.plot(rf_crosseval[col], 'r-^', label='RF')
    plt.title(f'不同模型的{col}对比')
    plt.xlabel('重复交叉验证次数')
    plt.ylabel(col,fontsize=16)
    plt.legend()
plt.tight_layout()
plt.show()


# In[55]:


plt.subplots(1,4,figsize=(16,3))
for i,col in enumerate(lgb_crosseval2.columns):
    n=int(str('14')+str(i+1))
    plt.subplot(n)
    plt.plot(lgb_crosseval2[col], 'k', label='LGB')
    plt.plot(xgb_crosseval2[col], 'b-.', label='XGB')
    plt.plot(rf_crosseval2[col], 'r-^', label='RF')
    plt.title(f'不同模型的{col}方差对比')
    plt.xlabel('重复交叉验证次数')
    plt.ylabel(col,fontsize=16)
    plt.legend()
plt.tight_layout()
plt.show()


# LGB的效果最好，选它进行搜参

# ### 搜超参数

# In[43]:


#利用K折交叉验证搜索最优超参数
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[46]:


# Choose best hyperparameters by RandomizedSearchCV
#随机搜索决策树的参数
param_distributions = {'max_depth': range(4, 10), 'subsample':np.linspace(0.5,1,5 ),'num_leaves': [15, 31, 63, 127],
                       'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}
                        # 'min_child_weight':np.linspace(0,0.1,2 ),
kfold = KFold(n_splits=3, shuffle=True, random_state=1)
model =RandomizedSearchCV(estimator= LGBMRegressor(objective='regression',random_state=0),
                          param_distributions=param_distributions, n_iter=200)
model.fit(X_train_s, y_train)


# In[47]:


model.best_params_


# In[48]:


model = model.best_estimator_
model.score(X_val_s, y_val)


# In[52]:


#网格化搜索学习率和估计器个数
param_grid={'learning_rate': np.linspace(0.05,0.3,6 ), 'n_estimators':[100,500,1000,1500, 2000]}
model =GridSearchCV(estimator= LGBMRegressor(objective='regression',random_state=0), param_grid=param_grid, cv=3)
model.fit(X_train_s, y_train)


# In[53]:


model.best_params_


# In[54]:


model = model.best_estimator_
model.score(X_val_s, y_val)


# In[129]:


#利用找出来的最优超参数在所有的训练集上训练，然后预测
model=LGBMRegressor(objective='regression',subsample=0.875,learning_rate= 0.05,n_estimators= 2500,num_leaves=127,
                    max_depth= 9,colsample_bytree=0.8,random_state=0)
model.fit(X_train_s, y_train)
model.score(X_val_s, y_val)


# ### 变量重要性排序图

# In[126]:


model=LGBMRegressor(objective='regression',subsample=0.875,learning_rate= 0.05,n_estimators= 2500,num_leaves=127,
                    max_depth= 9,colsample_bytree=0.8,random_state=0)
model.fit(data.to_numpy(),y.to_numpy())
model.score(data.to_numpy(), y.to_numpy())


# In[127]:


sorted_index = model.feature_importances_.argsort()
plt.barh(range(data.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(data.shape[1]), data.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting')


# ### 预测储存

# In[128]:


pred = model.predict(data2)
df=pd.DataFrame(ID)
df['price']=pred
df.to_csv('全部数据预测结果.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




