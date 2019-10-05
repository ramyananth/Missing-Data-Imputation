#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[17]:


data = pd.read_csv("train_with_missing/1.csv",index_col = False)


# In[18]:


values = data.values
X_filled_knn = KNN(k=2).fit_transform(values)
y = values[:,13]
x = np.delete(X_filled_knn, 0, 1)


# In[19]:


xtr,xt,ytr,yt=train_test_split(x, y, test_size=0.1, random_state=42)


# In[20]:


log = LogisticRegression()
xtr[np.isnan(xtr)] = 0
ytr[np.isnan(ytr)] = 0
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

encoded_ytr = lab_enc.fit_transform(ytr)

log.fit(xtr,encoded_ytr)


# In[21]:


ytp= log.predict(xt)


# In[23]:


from sklearn.metrics import mean_squared_error
print("LR MSE = " , mean_squared_error(yt, ytp)*100 , "%")

