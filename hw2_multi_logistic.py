#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# For reading the data you can use either numpy or pandas and accordingly handle your processing. An example could be
# my_data = np.genfromtxt('SPAM-HW1.csv', delimiter=',')
    


# In[2]:


def sigmoid(x):
    sig_x = 1.0/(1 + np.exp(-x))
    return sig_x


# In[59]:


def single_logistic_train(X_train, y_train, itermax=1000, eta = 1):
    dataMatrix = np.mat(X_train)
    labelMat = y_train
    m, n = dataMatrix.shape
    theta = np.ones((n, 1))
    for i in range(itermax):
        h = sigmoid(dataMatrix.dot(theta))
        error = h - labelMat
        theta = theta - eta * (dataMatrix.T * error)
    return np.asarray(theta)


# In[53]:


def single_logistic_train(X_train, y_train, itermax=1000, eta = 0.001, mini_len = 200):
    """
    This function should implement fitting or training your model in question. 
    """
    dataMatrix = np.mat(X_train)
    labelMat = y_train
    m, n = dataMatrix.shape
    theta = np.ones(n)
    for i in range(itermax):
        randIndex = int(np.random.uniform(0, m))
        for k in range(mini_len):
            h = sigmoid(np.dot(mnist_train_X[randIndex],theta))
            error = h -mnist_train_y[randIndex]
            theta = theta - eta * (error * mnist_train_X[randIndex])
    return theta


# In[60]:


def multi_logistic_train(X_train, y_train, itermax=100, eta = 1, mini_len = 500, class_num = 10):
    """
    This function should implement fitting or training your model in question. 
    """
    m, n = X_train.shape
    mnist_train_y_class = []
    for i in range(m):
        mnist_train_y_class.append([0]*class_num)
        mnist_train_y_class[i][y_train[i]] = 1
    mnist_train_y_mat = np.mat(mnist_train_y_class)
    theta = []
    for i in range(class_num):
        theta_tmp = single_logistic_train(X_train, mnist_train_y_mat[:,i], itermax, eta).reshape((n,1))
        theta.append(theta_tmp[:,0])
    return np.mat(theta)


# In[44]:


def multi_logistic_predict(X_valid,theta):
    """
    Here, using the trained model, implement how to predict when you just have feature vector. 
    """
    h=np.dot(mnist_train_X, theta.T)
    return h


# In[61]:


"""
Main - Here goes the overall logic.
"""
# cross-validation to get train and validation data
# We will use cross validation for training and validation. In this assignment, we will not use test split separately.
#  Let us say we want k-fold with k=5 - shuffle the data and partition into k-equal partitions
#  Save paritions into dictionaries
np.random.seed(5525)
mnist_train = pd.read_csv('mnist_train.csv',delimiter = ',',header = None)
mnist_test = pd.read_csv('mnist_test.csv',delimiter = ',',header = None)
mnist_train_X = mnist_train.iloc[:,1:]
mnist_train_X = np.array(mnist_train_X)
mnist_train_y = mnist_train.iloc[:,0]
mnist_train_y = np.array(mnist_train_y)
mnist_test_X = mnist_test.iloc[:,1:]
mnist_test_X = np.array(mnist_train_X)
mnist_test_y = mnist_test.iloc[:,0]
mnist_test_y = np.array(mnist_train_y)
theta = multi_logistic_train(mnist_train_X, mnist_train_y)
h=multi_logistic_predict(mnist_test_X,theta)
h_argmax= np.argmax(h, axis=1)


# In[57]:


print("The confusion matrix is:")
print(h)


# In[58]:


print("The accuracy is:")
print(sum(np.array(h_argmax).ravel() == mnist_train_y)/len(mnist_train_y))


# In[ ]:




