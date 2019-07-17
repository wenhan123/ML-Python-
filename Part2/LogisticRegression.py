#!/usr/bin/env python
# coding: utf-8

# 考虑二分类问题，只有0/1

# In[2]:


import numpy as np
from sklearn.datasets import load_iris,make_classification
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sigmoid = lambda x:1./(1+np.exp(-x))


# In[4]:


x = np.linspace(-10,10,10000)
y = sigmoid(x)
plt.plot(x,y)
plt.scatter(0,sigmoid(0))


# In[5]:


data = load_iris()
x_trian = data.data[data.target != 0]
y_trian = data.target[data.target !=0]
y_trian[y_trian==1] = 0
y_trian[y_trian==2] = 1
y_trian = y_trian.reshape(-1,1)
x_trian[:5],y_trian[:5]


# In[8]:


#定义对数回归的梯度下架解法
class LogisticRegression_1():
    def __init__(self):
        self.sigmoid = lambda x:1./(1+np.exp(-x))
    def fit(self,x,y,lr):
        self.w = np.random.randn(x.shape[1],1)
        self.lr = lr
        for _ in range(1000):
            y_pred = self.sigmoid(x @ self.w)
            self.w -=self.lr * x.T @ (y_pred-y)
            print(np.mean(0.5*(y_pred-y)**2))
    def predict(self,x):
        y_pred = np.round(self.sigmoid(x @ self.w))
        return y_pred


# In[9]:


LR = LogisticRegression_1()
LR.fit(x_trian,y_trian,0.01)


# In[10]:


y_pred = LR.predict(x_trian)
acc = np.sum(y_trian==y_pred,axis=0)/len(y_trian)
acc[0]


# 下面是较为复杂的一种实现

# In[1]:


from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[2]:


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    # print(data)
    return data[:,:2], data[:,-1]


# In[3]:


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[4]:



class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, X, y):
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose(
                    [data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.learning_rate, self.max_iter))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


# In[5]:


lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[6]:


x_ponits = np.arange(4, 8)
y_ = -(lr_clf.weights[1]*x_ponits + lr_clf.weights[0])/lr_clf.weights[2]
plt.plot(x_ponits, y_)

#lr_clf.show_graph()
plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()


# scikit-learn实例
# sklearn.linear_model.LogisticRegression
# solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：
# 
# a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
# b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
# c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
# d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。

# In[7]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)


# In[8]:


clf.fit(X_train, y_train)


# In[9]:


print(clf.coef_, clf.intercept_)


# In[10]:


x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


# In[ ]:




