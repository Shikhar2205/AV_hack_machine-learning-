# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:23:34 2018

@author: Shikhar
"""
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# # Basic steps to import the data 
# 
# We are taking the test and train dataset into the dataframe 
# Target variables are blah blah blah

# In[32]:


test=pd.read_csv('F:\\Kaggle kernels shikhar\\Big mart 2\\Test_u94Q5KV.csv')
train=pd.read_csv('F:\\Kaggle kernels shikhar\\Big mart 2\\Train_UWu5bXk.csv')
train1=pd.read_csv('F:\\Kaggle kernels shikhar\\Big mart 2\\Train_UWu5bXk.csv')
print (test.shape,train.shape)
print (train.info())
y=train.Item_Outlet_Sales            
train=train.drop(['Item_Outlet_Sales'],axis=1)
num=train.select_dtypes(exclude=['object'])
print (num.info())




# 
# 
# 
# ## Part 1: Handling the missing data 
# 

# In[33]:


print(train.isnull().sum())
train['Item_Weight']=train['Item_Weight'].fillna(train.Item_Weight.mean())
print (train['Outlet_Size'].value_counts())
train['Outlet_Size']=train['Outlet_Size'].fillna('Medium')
test['Item_Weight']=test['Item_Weight'].fillna(test.Item_Weight.mean())
test['Outlet_Size']=test['Outlet_Size'].fillna('Medium')


# Here we are mapping Low fat categories to low and regular to reg

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
fdata=[train,test]


print (train.Item_Fat_Content.unique())

train.Item_Fat_Content=train.Item_Fat_Content.map({'Low Fat':'low','Regular':'reg','LF':'low','low fat':'low','reg':'reg'})
train1.Item_Fat_Content=train1.Item_Fat_Content.map({'Low Fat':'low','Regular':'reg','LF':'low','low fat':'low','reg':'reg'})


print (train.Item_Fat_Content.unique())
print (train1.Item_Fat_Content.unique())



# In[139]:


g1=train1[['Item_Fat_Content','Item_Outlet_Sales']].groupby(['Item_Fat_Content'],as_index=False).mean()
print (g1)
f,(axis1,axis2)=plt.subplots(1,2,figsize=(10,6))
sns.barplot(x='Item_Fat_Content',y='Item_Outlet_Sales',data=g1,ax=axis1)


# In[159]:


print (train.info())
print (train.Item_Type.unique())
plot =plt.figure(figsize=(20,10))

g2=train1[['Item_Type','Item_Outlet_Sales']].groupby(['Item_Type'],as_index=False).mean()
plot =sns.barplot(x='Item_Type',y='Item_Outlet_Sales',data=g2)


# In[165]:


print (train.info())
print (train.Outlet_Identifier.unique())
plot =plt.figure(figsize=(20,10))

g3=train1[['Outlet_Identifier','Item_Outlet_Sales']].groupby(['Outlet_Identifier'],as_index=False).mean()
plot =sns.barplot(x='Outlet_Identifier',y='Item_Outlet_Sales',data=g3)


# In[166]:


print (train.info())
print (train.Outlet_Identifier.unique())
plot =plt.figure(figsize=(20,10))

g3=train1[['Outlet_Size','Item_Outlet_Sales']].groupby(['Outlet_Size'],as_index=False).mean()
plot =sns.barplot(x='Outlet_Size',y='Item_Outlet_Sales',data=g3)


# In[167]:


print (train.info())
print (train.Outlet_Identifier.unique())
plot =plt.figure(figsize=(20,10))

g3=train1[['Outlet_Type','Item_Outlet_Sales']].groupby(['Outlet_Type'],as_index=False).mean()
plot =sns.barplot(x='Outlet_Type',y='Item_Outlet_Sales',data=g3)


# In[169]:


print (train.info())
print (train.Outlet_Identifier.unique())
plot =plt.figure(figsize=(10,5))

g3=train1[['Outlet_Location_Type','Item_Outlet_Sales']].groupby(['Outlet_Location_Type'],as_index=False).mean()
plot =sns.barplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=g3)


# In[15]:




# In[36]:




test.Item_Fat_Content=test.Item_Fat_Content.map({'Low Fat':'low','Regular':'reg','LF':'low','low fat':'low','reg':'reg'})
train=pd.get_dummies(data=train,columns=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type'])
test=pd.get_dummies(data=test,columns=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type'])


# In[50]:


train=train.drop(['Item_Identifier'],axis=1)
test=test.drop(['Item_Identifier'],axis=1)
print (train.shape,test.shape)

# In[66]:


# In[67]:

from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, 
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'rbf',
    'C' : 0.025
    }

rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)


y_train = y.ravel()

x_train = train.values # Creates an array of the train data
x_test = test.values 
print (y_train)




#Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
print("Training is complete")


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test), axis=1)
import xgboost as xgb

gbm = xgb.XGBRegressor(
    #learning_rate = 0.02,
 n_estimators= 500,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
y_pred = gbm.predict(x_test)
print (y_pred)


test1=pd.read_csv('F:\\Kaggle kernels shikhar\\Big mart 2\\Test_u94Q5KV.csv')
print (test.info())
path='E:\\'
submit=pd.DataFrame()
submit['Item_Identifier'] = test1["Item_Identifier"]
submit['Outlet_Identifier'] = test1["Outlet_Identifier"]
submit['Item_Outlet_Sales']=y_pred
submit.to_csv(path+'greenl.csv',index=False)
print('complete')


