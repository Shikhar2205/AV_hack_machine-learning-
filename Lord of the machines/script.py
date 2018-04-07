# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:25:07 2018

@author: Shikhar
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

'''test=pd.read_csv('test1.csv')
train=pd.read_csv('train1.csv')
cam=pd.read_csv('campaign_data.csv')

all_data=pd.concat([train,test],axis=0)

final_data = all_data.merge(cam, on = 'campaign_id', copy = False)

#print (final_data.info())

y=train.iloc[:,4:6]
final_data.drop(['is_open','is_click','campaign_id','id'],axis=1,inplace=True)
from datetime import datetime
final_data['date'] = pd.to_datetime(final_data['send_date'], infer_datetime_format=True )

month = final_data['date'].dt.month
year  = final_data['date'].dt.year
day   = final_data['date'].dt.day
hour  = final_data['date'].dt.hour
final_data.drop(['send_date','email_url'],axis=1)
final_data.to_csv('this_data.csv',index=False)   '''

final=pd.read_csv('this_data.csv')
final.drop(['send_date','email_url'],axis=1,inplace=True)
print (final.info())
final['date_parsed'] = pd.to_datetime(final['date'], format = "%Y-%m-%d %H:%M:%S")
month = final['date_parsed'].dt.month
year  = final['date_parsed'].dt.year
day   = final['date_parsed'].dt.day
hour  = final['date_parsed'].dt.hour
final['year']=year
final['month']=month
final['day']=day
final.drop(['date','date_parsed','email_body'],axis=1,inplace=True)
num=final.select_dtypes(exclude=['object'])
test=pd.read_csv('test1.csv')
train=pd.read_csv('train1.csv')
num['click']=train['is_click']
num['open']=train['is_open']
'''cor=num.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(cor, vmax=.8, annot=True)
'''
y=train[['is_open','is_click']]
y['prob']=0
y['prob'].loc[(y['is_click']==1) &  (y['is_open']==1)]=6/6
y['prob'].loc[(y['is_click']==0) &  (y['is_open']==1)]=3/6
y['prob'].loc[(y['is_click']==0) &  (y['is_open']==0)]=0
y.drop(['is_click','is_open'],axis=1,inplace=True)



    
final=pd.get_dummies(data=final,columns=['communication_type','subject'])

train1=final[:1023191]
test1=final[1023191:]

#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
'''X_train, X_test, y_train, y_test = train_test_split(final, y, test_size=0.3, random_state=0)'''
'''from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(train1, y)

y_pred = logreg.predict(test1)'''
'''print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))'''
'''
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

'''
train=pd.read_csv('train1.csv')

y=y.as_matrix()
y=y.ravel()

from sklearn.svm import SVR

clf=SVR(kernel='rbf')
clf.fit(train1,y)
y_pred=clf.predict(test1)



for i in range(len(y_pred)):
    if (y_pred[i]>=0.5):
        y_pred[i]=1
    else:
        y_pred[i]=0
        




submit=pd.DataFrame()
submit['id'] = test["id"]
submit['is_click']=y_pred
submit.to_csv('submission33.csv',index=False)

print ('complete')

