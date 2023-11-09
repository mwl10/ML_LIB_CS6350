#!/usr/bin/env python
# coding: utf-8

# In[620]:


import numpy as np
np.random.seed(seed=1)
np.set_printoptions(precision=7)
import pandas as pd


# ### Process bank-note dataset

# In[621]:


train = pd.read_csv('../datasets/bank-note/train.csv', header=None).to_numpy()
test = pd.read_csv('../datasets/bank-note/test.csv', header=None).to_numpy()


# In[622]:


### last column is label (-1,1)

x_train = train[:,:-1]
y_train = train[:,-1]
y_train = np.where(y_train == 0, -1,1)

x_test = test[:,:-1]
y_test = test[:,-1]
y_test= np.where(y_test == 0, -1,1)

# add column of ones to wrap in b
x_train = np.concatenate((x_train, np.ones((x_train.shape[0],1))), axis=1)
x_test = np.concatenate((x_test, np.ones((x_test.shape[0],1))), axis=1)

shuffle = np.random.choice(len(x_train), len(x_train), replace=False)
x_train = x_train[shuffle]
y_train = y_train[shuffle]


# ### Standard Perceptron

# In[623]:


### utilities
calc_avg_error = lambda preds,labels: (preds != labels).mean()
predict = lambda x,w: np.sign(x @ w)


# In[624]:


# initialize w
w_1 = np.zeros((x_train.shape[1],))


# In[625]:


lr = 0.001
num_epochs = 10

for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        
        pred = predict(x_train[i], w_1)
        label = y_train[i]
        if pred != label:
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w_1 += lr * x_train[i]
            # if we guess positive but answer is negative, decrease
            else:
                w_1 -= lr * x_train[i]
        


# In[626]:


train_preds_1 = predict(x_train,w_1)
train_avg_error_1 = calc_avg_error(train_preds_1,y_train)

test_preds_1 = predict(x_test,w_1)
test_avg_error_1 = calc_avg_error(test_preds_1, y_test)

print(f'STANDARD PERCEPTRON \n {w_1=},\n {train_avg_error_1=},\n {test_avg_error_1=}')


# ### Voted Perceptron

# In[627]:


### utilities
calc_avg_error = lambda preds,labels: (preds != labels).mean()
predict = lambda x,w: np.sign(x @ w)


# In[628]:


# initialize w
w_2 = np.zeros((x_train.shape[1],))


# In[629]:


lr = 0.001
num_epochs = 10

## have a count of number of correct answers for a w, before it is wrong and thus is updated 

### count number of correct 
c=1
keeps = []
for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        pred = predict(x_train[i], w_2)
        label = y_train[i]
        if pred != label:
            keeps.append((w_2,c))
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w_2 = w_2 + (lr * x_train[i])
            # if we guess positive but answer is negative, decrease
            else:
                w_2 = w_2 - (lr * x_train[i])
        
            c = 1 
        else:
            c += 1

def voted_predict(x,keeps):
    sum = np.array([c * predict(x,w) for (w,c) in keeps]).sum(axis=0)
    return np.sign(sum)
     


# In[630]:


train_preds_2 = voted_predict(x_train,keeps)
train_avg_error_2 = calc_avg_error(train_preds_2,y_train)

test_preds_2 = voted_predict(x_test,keeps)
test_avg_error_2 = calc_avg_error(test_preds_2, y_test)

print(f'VOTED PERCEPTRON \n {train_avg_error_2=},\n {test_avg_error_2=} \n {len(keeps)} distinct classifiers:')
for (w,c) in keeps:
    print(f'{w=}, {c=} \\\ ')


# ### Averaged Perceptron

# In[631]:


### utilities
calc_avg_error = lambda preds,labels: (preds != labels).mean()
predict = lambda x,w: np.sign(x @ w)


# In[632]:


# initialize w
w_3 = np.zeros((x_train.shape[1],))


# In[633]:


lr = 0.001
num_epochs = 10
a = np.zeros_like(w_3)
for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        
        pred = predict(x_train[i], w_3)
        label = y_train[i]
        if pred != label:
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w_3 += lr * x_train[i]
            # if we guess positive but answer is negative, decrease
            else:
                w_3 -= lr * x_train[i]
        a = a + w_3
        
# a = a / (num_epochs * x_train.shape[0])


# In[634]:


train_preds_3 = predict(x_train,a)
train_avg_error_3 = calc_avg_error(train_preds_3,y_train)

test_preds_3 = predict(x_test,a)
test_avg_error_3 = calc_avg_error(test_preds_3, y_test)

print(f'AVERGAGED PERCEPTRON \n {w_3=}\n {a=}\n {train_avg_error_3=},\n {test_avg_error_3=}')


# In[ ]:




