#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from fancyimpute import IterativeSVD
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# In[5]:


data = pd.read_csv("train_with_missing/1.csv",index_col = False)
data.head(5)


# In[6]:


data.isnull().sum(axis = 0)


# In[7]:


values = data.values
values.shape


# In[9]:


X_filled_svd = IterativeSVD().fit_transform(values)


# In[12]:


X_filled_svd


# In[42]:


y = values[:,13]
x = np.delete(X_filled_svd, 0, 1)


# In[43]:


xtr,xt,ytr,yt=train_test_split(x, y, test_size=0.1, random_state=42)


# In[48]:


clf = svm.SVC(gamma='scale')
X = np.array(xtr)
y = np.array(ytr)
from sklearn.svm import SVC
clf = SVC(gamma='auto')
X[np.isnan(X)] = 0
y[np.isnan(y)] = 0


# In[73]:


from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

encoded_y = lab_enc.fit_transform(y)

print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y.astype('int')))
print(utils.multiclass.type_of_target(encoded))


# In[74]:


clf.fit(X, encoded)


# In[75]:


ytp = clf.predict(xt)
print(ytp)


# In[76]:


from sklearn.metrics import mean_squared_error
print("SVM MSE = " , mean_squared_error(yt, ytp)*100 , "%")


# In[77]:


xtr[np.isnan(xtr)] = 0
ytr[np.isnan(ytr)] = 0
encoded_ytr = lab_enc.fit_transform(y)


gnb = GaussianNB()
gnb.fit(xtr,encoded_ytr)
ytp = gnb.predict(xt)
print("Naive bayes MSE = " , mean_squared_error(yt, ytp)*100 , "%")

