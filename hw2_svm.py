#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt


# In[348]:


hw2data_raw_data = pd.read_csv('hw2data.csv',delimiter = ',',header = None)
hw2data_raw_data.columns = ['x_1', 'x_2',"y"]
np.random.seed(5525)
reo_list=list(hw2data_raw_data.index)
np.random.shuffle(reo_list)
ave_len_5 = int(len(reo_list)/5)
testing_data = hw2data_raw_data.iloc[reo_list[0:ave_len_5]]
remain_data = hw2data_raw_data.iloc[reo_list[ave_len_5:5*ave_len_5]]

reo_list_2=list(remain_data.index)
ave_len_10 = int(len(reo_list_2)/10)
index_1 = reo_list_2[0:ave_len_10]
index_2 = reo_list_2[ave_len_10:2*ave_len_10]
index_3 = reo_list_2[2*ave_len_10:3*ave_len_10]
index_4 = reo_list_2[3*ave_len_10:4*ave_len_10]
index_5 = reo_list_2[4*ave_len_10:5*ave_len_10]
index_6 = reo_list_2[5*ave_len_10:6*ave_len_10]
index_7 = reo_list_2[6*ave_len_10:7*ave_len_10]
index_8 = reo_list_2[7*ave_len_10:8*ave_len_10]
index_9 = reo_list_2[8*ave_len_10:9*ave_len_10]
index_10 = reo_list_2[9*ave_len_10:10*ave_len_10]

X_shuffled = {0:hw2data_raw_data.iloc[index_1].loc[:,["x_1","x_2"]],1:hw2data_raw_data.iloc[index_2].loc[:,["x_1","x_2"]],2:hw2data_raw_data.iloc[index_3].loc[:,["x_1","x_2"]],3:hw2data_raw_data.iloc[index_4].loc[:,["x_1","x_2"]],4:hw2data_raw_data.iloc[index_5].loc[:,["x_1","x_2"]],+
              5:hw2data_raw_data.iloc[index_6].loc[:,["x_1","x_2"]],6:hw2data_raw_data.iloc[index_7].loc[:,["x_1","x_2"]],7:hw2data_raw_data.iloc[index_8].loc[:,["x_1","x_2"]],8:hw2data_raw_data.iloc[index_9].loc[:,["x_1","x_2"]],9:hw2data_raw_data.iloc[index_10].loc[:,["x_1","x_2"]]}
y_shuffled = {0:hw2data_raw_data.iloc[index_1].loc[:,"y"],1:hw2data_raw_data.iloc[index_2].loc[:,"y"],2:hw2data_raw_data.iloc[index_3].loc[:,"y"],3:hw2data_raw_data.iloc[index_4].loc[:,"y"],4:hw2data_raw_data.iloc[index_5].loc[:,"y"],+
             5:hw2data_raw_data.iloc[index_6].loc[:,"y"],6:hw2data_raw_data.iloc[index_7].loc[:,"y"],7:hw2data_raw_data.iloc[index_8].loc[:,"y"],8:hw2data_raw_data.iloc[index_9].loc[:,"y"],9:hw2data_raw_data.iloc[index_10].loc[:,"y"]}
X_test = remain_data.loc[:,["x_1","x_2"]]
y_test = remain_data.loc[:,"y"]


# In[336]:


def get_next_train_valid(X_shuffled, y_shuffled, itr):
    """
    - itr value can tell you which fold you pick up for valid and rest go to training
    - use 1 fold for validation and all other folds for training
    - in next iteration, use a different fold for validation and remaining folds for training
        
    Basically, here you will determine which fold goes for validation and which ones for training
    """
    X_valid = pd.DataFrame(X_shuffled.get(itr))
    y_valid = pd.DataFrame(y_shuffled.get(itr))
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    for i in range(10):
        if (i != itr):
            X_train = pd.concat([X_train,X_shuffled.get(i)],axis = 0)
            y_train = pd.concat([y_train,y_shuffled.get(i)],axis = 0)
    return(X_train,y_train,X_valid,y_valid)
    #return training and validation data


# In[364]:


def svmfit(X_train, y_train, bigC = 0.01):
    """
    This function should implement fitting or training your model in question. 
    """
    rnum = X_train.shape[0]
    cnum = X_train.shape[1]
    tmp_K = np.zeros((rnum, rnum))
    P = np.dot(y_train * X_train*1.0, (y_train * X_train*1.0).T)
    q = -1 * np.ones((rnum,1))
    G = np.vstack((np.eye(rnum)* -1 ,np.eye(rnum)))
    h = np.hstack((np.zeros(rnum), np.ones(rnum) * bigC))
    A = y_train.T
    b = np.array([0.0])
    
    P_cvm = cvxopt.matrix(P)
    q_cvm = cvxopt.matrix(q)
    G_cvm = cvxopt.matrix(G)
    h_cvm = cvxopt.matrix(h)
    A_cvm = cvxopt.matrix(A)
    b_cvm = cvxopt.matrix(b)
    sol=cvxopt.solvers.qp(P_cvm, q_cvm, G_cvm, h_cvm, A_cvm, b_cvm,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    lamb = np.array(sol['x'])
    w = np.zeros(cnum,)
    for i in range(0, rnum):
        if (sol["x"][i]>=0 and sol["x"][i]<=bigC):
            w = w + np.multiply(y_t[i][0]*sol["x"][i],list(X_t[i]))
    S = (lamb > 1e-5).flatten()
    b = y_train[S] - np.dot(X_train[S], w)
    model = (w,b[0][0])
    return model
    # return model


# In[291]:


def predict(X_valid, model):
    """
    Here, using the trained model, implement how to predict when you just have feature vector. 
    """
    w = model[0]
    b = model[1]
    y_predicted = np.sign(np.dot(X_valid,w) + b)
    return y_predicted
    # return 


# In[385]:


"""
Main - Here goes the overall logic.
"""
# cross-validation to get train and validation data
# We will use cross validation for training and validation. In this assignment, we will not use test split separately.
#  Let us say we want k-fold with k=5 - shuffle the data and partition into k-equal partitions
#  Save paritions into dictionaries

#Loop through 5 times, each time selecting 1 fold as validation and remaining as train data sets using function get_next_train_valid 
#   train your model
#   predict target on validation and training
#   compute error i.e. RMSE or classification accuracy as per assignment question
#
#Plot or print as per question - Mean RMSE or accuracy

Training_Error_Rate_list = []
Validation_Error_Rate_list = []
#Testing_Error_Rate_list = []
model_list = []
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
for itr in range(10):
    X_t = get_next_train_valid(X_shuffled, y_shuffled, itr)[0]
    X_t = np.array(X_t)
    y_t = get_next_train_valid(X_shuffled, y_shuffled, itr)[1]
    y_t = np.array(y_t)
    X_v = get_next_train_valid(X_shuffled, y_shuffled, itr)[2]
    X_v = np.array(X_v)
    y_v = get_next_train_valid(X_shuffled, y_shuffled, itr)[3]
    y_v = np.array(y_v)  
    model = svmfit(X_t,y_t)
    model_list.append(model)
    y_p_t = predict(X_t,model)
    ero_t = 1-(sum(y_p_t == y_t.T[0])/y_p_t.shape[0])
    y_p_v = predict(X_v,model)
    ero_v = 1-(sum(y_p_v == y_v.T[0])/y_p_v.shape[0])
    Training_Error_Rate_list.append(ero_t)
    Validation_Error_Rate_list.append(ero_v)   
traning_mean = np.mean(Training_Error_Rate_list)
validation_mean = np.mean(Validation_Error_Rate_list)
min_ind = np.argmin(Validation_Error_Rate_list)
chosen_model = model_list[min_ind]
y_t_t = predict(X_test,chosen_model)
ero_test = 1-(sum(y_t_t == (np.array(y_test).T)[0])/y_t_t.shape[0])


# In[386]:


ero_test


# In[481]:


plot_data_x = hw2data_raw_data['x_1']
plot_data_y = hw2data_raw_data['x_2']
plot_data_c = hw2data_raw_data['y']
plot_data_c[plot_data_c == 1] = 'r'
plot_data_c[plot_data_c == -1] = 'b'


# In[482]:


plt.scatter(plot_data_x, plot_data_y,color = plot_data_c)


# In[390]:


C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_traning_mean = []
C_validation_mean = []
C_Testing_Error = []

for i in range(8):
    Training_Error_Rate_list = []
    Validation_Error_Rate_list = []
    model_list = []
    for itr in range(10):
        X_t = get_next_train_valid(X_shuffled, y_shuffled, itr)[0]
        X_t = np.array(X_t)
        y_t = get_next_train_valid(X_shuffled, y_shuffled, itr)[1]
        y_t = np.array(y_t)
        X_v = get_next_train_valid(X_shuffled, y_shuffled, itr)[2]
        X_v = np.array(X_v)
        y_v = get_next_train_valid(X_shuffled, y_shuffled, itr)[3]
        y_v = np.array(y_v)  
        model = svmfit(X_t,y_t,C_list[i])
        model_list.append(model)
        y_p_t = predict(X_t,model)
        ero_t = 1-(sum(y_p_t == y_t.T[0])/y_p_t.shape[0])
        y_p_v = predict(X_v,model)
        ero_v = 1-(sum(y_p_v == y_v.T[0])/y_p_v.shape[0])
        Training_Error_Rate_list.append(ero_t)
        Validation_Error_Rate_list.append(ero_v)   
    traning_mean = np.mean(Training_Error_Rate_list)
    validation_mean = np.mean(Validation_Error_Rate_list)
    min_ind = np.argmin(Validation_Error_Rate_list)
    chosen_model = model_list[min_ind]
    y_t_t = predict(X_test,chosen_model)
    Testing_Error = 1-(sum(y_t_t == (np.array(y_test).T)[0])/y_t_t.shape[0])
    C_traning_mean.append(traning_mean)
    C_validation_mean.append(validation_mean)
    C_Testing_Error.append(Testing_Error)


# In[392]:


plt.figure(figsize=(16,10))
plt.plot(np.log(C_list), C_traning_mean, marker='', linewidth = 2, label = 'Train')
plt.plot(np.log(C_list), C_validation_mean, marker='', linewidth = 2, label = 'Valid')
plt.plot(np.log(C_list), C_Testing_Error, marker='', linewidth = 2, label = 'Test')
plt.legend(fontsize=20)
plt.xlabel('C', fontsize=20)
plt.ylabel('error rate', fontsize=20)
plt.xticks(np.log(C_list), ['0.0001', '0.001', '0.01', '0.1','1', '10', '100', '1000' ], fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[ ]:




