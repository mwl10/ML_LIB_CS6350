#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np
import pandas as pd


# In[192]:


train = pd.read_csv('../datasets/bank-note/train.csv', header=None).to_numpy()
test = pd.read_csv('../datasets/bank-note/test.csv', header=None).to_numpy()


# In[193]:


### last column is label (-1,1)

x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]

# add column of ones to wrap in b
x_train = np.concatenate((x_train, np.ones((x_train.shape[0],1))), axis=1)
x_test = np.concatenate((x_test, np.ones((x_test.shape[0],1))), axis=1)

shuffle = np.random.choice(len(x_train), len(x_train), replace=False)
x_train = x_train[shuffle]
y_train = y_train[shuffle]


# ### Standard Perceptron

# In[194]:


calc_avg_error = lambda preds,labels: (preds != labels).mean()
predict = lambda x,w: np.sign(x @ w)


# In[195]:


# initialize w
w = np.zeros((x_train.shape[1],))


# In[196]:


lr = 0.001
num_epochs = 10

for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        
        pred = predict(x_train[i], w)
        label = y_train[i]
        if pred != label:
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w += lr * x_train[i]
            # if we guess positive but answer is negative, decrease
            else:
                w -= lr * x_train[i]
        


# In[197]:


train_preds = predict(x_train,w)
train_avg_error = calc_avg_error(train_preds,y_train)

test_preds = predict(x_test,w)
test_avg_error = calc_avg_error(test_preds, y_test)

print(f'STANDARD PERCEPTRON \n {w=},\n {train_avg_error=},\n {test_avg_error=}')


# ### Voted Perceptron

# In[198]:


# initialize w
w = np.zeros((x_train.shape[1],))


# In[199]:


lr = 0.001
num_epochs = 10

## have a count of number of correct answers for a w, before it is wrong and thus is updated 

### count number of correct 
c=0
keeps = []
for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        
        pred = predict(x_train[i], w)
        label = y_train[i]
        if pred != label:
            keeps.append((w,c))
            c = 0
            
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w += lr * x_train[i]
            # if we guess positive but answer is negative, decrease
            else:
                w -= lr * x_train[i]
            
        else:
            c += 1


def voted_predict(x,keeps):
    sum = np.array([c * predict(x,w) for (w,c) in keeps]).sum(axis=0)
    return np.sign(sum)


# In[200]:


train_preds = voted_predict(x_train,keeps)
train_avg_error = calc_avg_error(train_preds,y_train)

test_preds = voted_predict(x_test,keeps)
test_avg_error = calc_avg_error(test_preds, y_test)

print(f'VOTED PERCEPTRON \n {train_avg_error=},\n {test_avg_error=} \n KEEPS')
[print(f'{w=}, {c=}') for (w,c) in keeps[:10]]


# ### Averaged Perceptron

# In[201]:


# initialize w
w = np.zeros((x_train.shape[1],))


# In[202]:


lr = 0.001
num_epochs = 10
a = np.zeros_like(w)
for epoch in range(num_epochs):
    for i in range(x_train.shape[0]):
        
        pred = predict(x_train[i], w)
        label = y_train[i]
        if pred != label:
            # if we guess negative but answer is positive, increase
            if pred == -1:
                w += lr * x_train[i]
            # if we guess positive but answer is negative, decrease
            else:
                w -= lr * x_train[i]
        a+=w
        
a = a / (num_epochs * x_train.shape[0])


# In[203]:


train_preds = predict(x_train,a)
train_avg_error = calc_avg_error(train_preds,y_train)

test_preds = predict(x_test,a)
test_avg_error = calc_avg_error(test_preds, y_test)

print(f'AVERGAGED PERCEPTRON \n {train_avg_error=},\n {test_avg_error=} \n KEEPS')



# In[ ]:





# In[ ]:




