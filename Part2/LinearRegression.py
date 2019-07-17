#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

get_ipython().run_line_magic('matplotlib', 'inline')


# 构造数据集

# In[4]:


x_train,y_train = make_regression(n_samples=100,noise=20,n_features=1)
plt.scatter(x_train,y_train)


# In[30]:


#使用梯度下降来求解
class LinearRegression():
    def __init__(self):
        pass
    def fit(self,x,y,lr):
        x = np.insert(x,0,1,axis=1)
        y = y.reshape(-1,1)
        self.w = np.random.randn(x.shape[1],1)
        self.lr = lr
        
        for _ in range(50):
            y_pred = x @ self.w
            mse = np.mean(0.5*(y_pred-y)**2)
            grad_w = x.T@(y_pred-y)
            self.w -= self.lr*grad_w
            print(_,mse,self.w[0][0],self.w[1][0])
    def predict(self,x):
        x = np.insert(x,0,1,axis=1)
        return x @ self.w


# In[31]:


LR = LinearRegression()


# In[32]:


LR.fit(x_train,y_train,0.01)


# In[33]:


y_pred = LR.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,y_pred,'r--')


# In[ ]:




