#!/usr/bin/env python
# coding: utf-8

# In[572]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Matthew Lowery

# ### Load Concrete Dataset

# In[573]:


# load dataset 
train_fp =  '../datasets/concrete/train.csv'
test_fp = '../datasets/concrete/test.csv'

train_data = pd.read_csv(train_fp, header=None).to_numpy()
test_data =  pd.read_csv(test_fp, header=None).to_numpy()

X,y = train_data[:,:-1], train_data[:,-1]
X_test, y_test= test_data[:,:-1], test_data[:,-1]


# In[574]:


np.random.seed(seed=1)


# In[575]:


W = np.zeros((X.shape[1]))


# ### Batch Gradient Descent

# In[576]:


epochs = 20000
train_losses = np.zeros((epochs))
test_losses = np.zeros((epochs))
lr = 0.01
calc_loss = lambda y, y_pred: 0.5*((y-y_pred)**2).sum()
calc_dW = lambda y,y_pred, X: - X.T @ (y-y_pred)

for epoch in range(epochs):
    
    y_pred = X @ W 
    train_losses[epoch] = calc_loss(y,y_pred)
    dW = calc_dW(y,y_pred, X)
    norm_dW = np.sqrt((dW**2).sum())
    if norm_dW <= (10 ** -6):
        print(f'converged')
        converge_epoch = epoch
        break
    W -= lr * dW
    if epoch % 1000 == 0:
        print(f'{epoch=}, {train_losses[epoch]=}')


# In[577]:


plt.plot(train_losses[:converge_epoch])
plt.xlabel('epoch'), plt.ylabel('cost'), plt.title('batch gradient descent')
plt.savefig('batch_gradient_descent_loss_curve')


# In[578]:


y_pred_test = X_test @ W
y_pred_train = X @ W
print(f'batch GD final weight vector {W}')
print(f'batch GD final train_loss = {calc_loss(y,y_pred_train)}')
print(f'batch GD final test_loss = {calc_loss(y_test,y_pred_test)}')


# ## Stochastic Gradient Descent

# In[579]:


W = np.zeros((X.shape[1]))


# In[580]:


epochs = 35000
train_losses = np.zeros((epochs))
lr = 0.001

for epoch in range(epochs): # amount of times we go through entire dataset 
    i = np.random.choice(len(X))

    y_pred = X @ W 
    train_losses[epoch] = calc_loss(y,y_pred)
    # single example gradient
    dW = calc_dW(y[None,i],X[i] @ W, X[None,i]) 
    W -= lr * dW
    
    if epoch % 2000 == 0:
        print(f'{epoch=}, {train_losses[epoch]=}')
        
    


# In[581]:


plt.plot(train_losses)
plt.xlabel('epoch'), plt.ylabel('cost'), plt.title('stochastic gradient descent')
plt.savefig('stochastic_gradient_descent_loss_curve')


# In[582]:


y_pred_test = X_test @ W
y_pred_train = X @ W
print(f'stochastic GD final weight vector {W}')
print(f'stochastic GD final train_loss = {calc_loss(y,y_pred_train)}')
print(f'stochastic GD final test_loss = {calc_loss(y_test,y_pred_test)}')


# ### Analytical Solution

# In[583]:


W = np.linalg.inv(X.T @ X) @ X.T @ y


# In[584]:


y_pred_train = X @ W
y_pred_test = X_test @ W

print(f'analytical soln weight vector {W}')
print(f'analytical soln train_loss = {calc_loss(y,y_pred_train)}')
print(f'analytical soln test_loss = {calc_loss(y_test,y_pred_test)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Problem number 5 in paper problems

# In[585]:


data = np.array([[1,-1,2,1],[1,1,3,4],[-1,1,0,-1], [1,2,-4,-2],[3,-1,-1,0]])

X = data[:,:-1]
X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
y = data[:,-1]


# #### 5b. calc grad w/ W = [-1, 1, -1,b=-1]

# In[586]:


W = np.array([-1, 1, -1,-1])[:,None]
(1/X.shape[0]) * X.T @ (y[:,None]-(X @ W)).flatten()


# ### 5c. Analytical Soln

# In[587]:


np.linalg.inv(X.T @ X) @ X.T @ y


# ### 5d. stochastic gradient descent

# In[588]:


W = np.zeros((X.shape[1]))
epochs = 100
train_losses = np.zeros((epochs))
lr = 0.1
# print('initial weights', W)
for epoch in range(epochs): # amount of times we go through entire dataset 
    i = epoch % X.shape[0]
    y_pred = X @ W 
    train_losses[epoch] = calc_loss(y,y_pred)
    # single example gradient
    dW = calc_dW(y[None,i],X[i] @ W, X[None,i]) 
    W -= lr * dW
    # print('dW', dW)
    # print('W', W)
    # if epoch <= 4:
    #     print(f'{epoch=}, {train_losses[epoch]=}')


# In[589]:


plt.plot(train_losses)


# In[590]:


W


# In[ ]:




