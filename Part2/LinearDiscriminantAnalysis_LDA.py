#!/usr/bin/env python
# coding: utf-8

# 将样例投影到一条直线上，使得同类的样例投影点尽可能小，不同类投影点尽可能远离

# In[1]:


import os
import sys
import numpy as np
import operator
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


group1 = np.random.random((8,2))*5+20
group2 = np.random.random((8,2))*5+2


# In[4]:


x1 = group1
y1 = np.ones((8,1))
x0 = group2
y0 = np.zeros((8,1))


# In[6]:


plt.scatter(x1[:,0],x1[:,1],c = 'r')
plt.scatter(x0[:,0],x0[:,1],c = 'g')


# In[7]:


mean1 = np.array([np.mean(x1[:,0]),np.mean(x1[:,1])])
mean0 = np.array([np.mean(x0[:,0]),np.mean(x0[:,1])])
from numpy import mat
m1 = np.shape(x1)[0]
sw = np.zeros(shape=(2,2))
for i in range(m1):
    xsmean = mat(x1[i,:]-mean1)
    sw+=xsmean.transpose()*xsmean
m0 = np.shape(x0)[0]
for i in range(m0):
    xsmean = mat(x0[i,:]-mean0)
    sw+=xsmean.transpose()*xsmean
w = (mean0-mean1)*(mat(sw).I)


# In[9]:


w


# In[11]:


plt.scatter(x1[:,0],x1[:,1],c = 'r')
plt.scatter(x0[:,0],x0[:,1],c = 'g')
x = np.arange(0,25,0.1)
y = np.array((-w[0,0]*x)/w[0,1])
plt.plot(x,y)


# In[ ]:




