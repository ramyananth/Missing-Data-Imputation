#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler


# In[6]:


data = pd.read_csv("train_with_missing/1.csv",index_col = False)


# In[8]:


values = data.values
X_filled_softimpute = SoftImpute().fit_transform(values)


# In[9]:


X_filled_softimpute


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

y = values[:,13]
x = np.delete(X_filled_softimpute, 0, 1)
xtr,xt,ytr,yt=train_test_split(x, y, test_size=0.1, random_state=42)


# In[27]:


xtr[np.isnan(xtr)] = 0
ytr[np.isnan(ytr)] = 0

from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

encoded_ytr = lab_enc.fit_transform(ytr)

from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(xtr,ytr)


# In[28]:


ytp=clf.predict(xt)


# In[33]:


from sklearn.metrics import mean_squared_error
print("Decision tree MSE = " , mean_squared_error(yt, ytp)*100 , "%")


# In[32]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(xtr, ytr) 
ytp = neigh.predict(xt)
print("NB MSE = " , mean_squared_error(yt, ytp)*100 , "%")

