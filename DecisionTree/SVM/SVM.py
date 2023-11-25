#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')


# ### Loading Bank Note Dataset

# In[3]:


train = pd.read_csv('../datasets/bank-note/train.csv', header=None).to_numpy()
test = pd.read_csv('../datasets/bank-note/test.csv', header=None).to_numpy()


# In[145]:


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


# ### Problem 2 SVM Primal

# In[149]:


### utilities
calc_avg_error = lambda preds,labels: (preds != labels).mean()
predict = lambda x,w: np.sign(x @ w)

def calc_avg_j(w_0,w,C,N,X,labels):
    preds = X @ w
    margins = labels*preds
    max_terms = np.max((np.zeros((len(margins))), 1-margins))
    j = 0.5*(w_0 @ w_0.T) + (C * N*max_terms)
    return j.mean()
    
print('Beginning SVM primal domain, stochastic sub-gradient descent')


# In[150]:


update_lr1 = lambda lr, t, a: lr / (1 + (lr/a)*t)
update_lr2 = lambda lr, t: lr / (1 + t)


# ### PROBLEM 2A, LR Schedule 1

# In[151]:


lr = 0.001
a = 0.5 ## need to tune this probs
num_epochs = 5000
Cs = [100/873, 500 / 873, 700 / 873]
N = len(x_train)
np.random.seed(42)
train_losses = np.zeros((len(Cs), num_epochs))
test_losses = np.zeros((len(Cs), num_epochs))

for c_i, C in enumerate(Cs):
    # initialize w
    w_0 = np.zeros((x_train.shape[1]-1,))
    w = np.concatenate((w_0,[0]))
    for epoch in range(num_epochs):
        ## shuffle at each epoch
        shuffle = np.random.choice(len(x_train), len(x_train), replace=False)
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        
        for i in range(x_train.shape[0]):
            ## shuffle at each epoch
            label = y_train[i]
            pred = x_train[i] @ w
            margin = label*pred
            max_term = np.max((0.0, 1-margin))
            if max_term: ### if margin is smaller than 1
                grad = np.concatenate((w_0,[0])) - (C*N*label*x_train[i])
                w = w - lr * grad
            else: ### if margin greater or equal to 1
                w_0 = w_0 - (lr * w_0)
    
        train_losses[c_i,epoch] = calc_avg_j(w_0,w,C,N,x_train,y_train)
        test_losses[c_i, epoch] = calc_avg_j(w_0,w,C,N,x_test,y_test)
        lr = update_lr1(lr, epoch, a)
    print(f'with {C=} and learning rate schedule 1:')
    print(f'final train loss = {train_losses[c_i,-1]}')
    print(f'final test loss = {test_losses[c_i,-1]}')
    train_acc = (predict(x_train,w) == y_train).mean()
    test_acc = (predict(x_test,w) == y_test).mean()
    print(f'{train_acc=}, {test_acc=}')
    print(f'{w=}')


# In[98]:


fig, ax = plt.subplots(len(Cs))
for i in range(len(Cs)):
    C = Cs[i]
    ax[i].plot(train_losses[i])
    ax[i].set_ylabel(f'{C=:.2f}',labelpad=0.4)
fig.suptitle('Primal SVM Training w/ Stochastic SubGD & LR 2')
fig.supylabel('LOSS')
ax[-1].set_xlabel('epoch')


# ### PROBLEM 2B, LR Schedule 2

# In[154]:


lr = 0.001
a = 0.5 ## need to tune this probs
num_epochs = 5000
Cs = [100/873, 500 / 873, 700 / 873]
N = len(x_train)
np.random.seed(42)
train_losses = np.zeros((len(Cs), num_epochs))
test_losses = np.zeros((len(Cs), num_epochs))

for c_i, C in enumerate(Cs):
    # initialize w
    w_0 = np.zeros((x_train.shape[1]-1,))
    w = np.concatenate((w_0,[0]))
    for epoch in range(num_epochs):
        ## shuffle at each epoch
        shuffle = np.random.choice(len(x_train), len(x_train), replace=False)
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        
        for i in range(x_train.shape[0]):
            ## shuffle at each epoch
            label = y_train[i]
            pred = x_train[i] @ w
            margin = label*pred
            max_term = np.max((0.0, 1-margin))
            if max_term: ### if margin is smaller than 1
                grad = np.concatenate((w_0,[0])) - (C*N*label*x_train[i])
                w = w - lr * grad
            else: ### if margin greater or equal to 1
                w_0 = w_0 - (lr * w_0)
    
        train_losses[c_i,epoch] = calc_avg_j(w_0,w,C,N,x_train,y_train)
        test_losses[c_i, epoch] = calc_avg_j(w_0,w,C,N,x_test,y_test)
        lr = update_lr2(lr, epoch)
    print(f'with {C=} and learning rate schedule 2:')
    print(f'final train loss = {train_losses[c_i,-1]}')
    print(f'final test loss = {test_losses[c_i,-1]}')
    train_acc = (predict(x_train,w) == y_train).mean()
    test_acc = (predict(x_test,w) == y_test).mean()
    print(f'{train_acc=}, {test_acc=}')
    print(f'{w=}')


# In[97]:


fig, ax = plt.subplots(len(Cs))
for i in range(len(Cs)):
    C = Cs[i]
    ax[i].plot(train_losses[i])
    ax[i].set_ylabel(f'{C=:.2f}',labelpad=0.4)
fig.suptitle('Primal SVM Training w/ Stochastic SubGD & LR 2')
fig.supylabel('LOSS')
ax[-1].set_xlabel('epoch')


# ### Homework problem 5

# In[18]:


print('Beginning HW written problem 1')

data = np.array([[0.5,-1,0.3,1], [-1,-2,-2,-1], [1.5,0.2,-2.5,1]])
x = data[:,:3]
y = data[:,-1]
lrs = [0.01, 0.005, 0.0025]
### wrap in b
x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)
w_0 = np.zeros((x.shape[1]-1,))
w = np.zeros((x.shape[1],))
N = len(x)
C = 1
### 3 steps w/ given learning rate
for i,lr in enumerate(lrs):
    label = y[i]
     # np.sign(x @ w)
    pred = x[i] @ w
    margin = label*pred
    max_term = np.max((0, 1-margin)) 
    if max_term: ### if margin is smaller than 1
        grad = np.concatenate((w_0,[0])) - (C*N*label*x[i])
        print(f'subgradient for step {i} is {grad}')
        w = w - lr * grad
    else: ### if margin greater or equal to 1
        print(f'subgradient for step {i} is {np.concatenate((w_0,[0]))=}')
        w_0 = w_0 - (lr * w_0)


# ### SVM Dual Domain

# In[178]:


from scipy.optimize import minimize 
from typing import Callable

def gaussian_kernel(X1,X2,gamma=0.1): 
    norm_diff_squared = ((X1 - X2)**2).sum(axis=-1)
    return np.exp(-1 * (norm_diff_squared / gamma))
    
def identity_kernel(X1,X2):
    return (X1 * X2).sum(axis=-1)

def objective(a,C,X,y,kernel: Callable):
    indexes = np.arange(len(X))
    i_s,j_s = np.meshgrid(indexes, indexes)
    term1 = y[i_s] * y[j_s] * a[i_s] * a[j_s] \
            * kernel(X[i_s],X[j_s])
    term1 = 0.5 * term1.sum()
    term2 = a.sum()
    return term1 - term2

y = y_train

constraints = ({'type': 'ineq','fun': lambda a: a },
               {'type': 'ineq', 'args': (C,),'fun': lambda a,C: C-a},
               {'type': 'eq', 'args': (y,),'fun': lambda a,y: (a*y).sum()}
              )


# ### Identity Kernel/ No Kernel w/ different Cs

# In[185]:


print('starting SVM duel with various C values and no kernel')

Cs = [100/873, 500 / 873, 700 / 873]
optims = []
for C in Cs:
    y = y_train
    a = np.zeros((len(x_train))) 
    constraints = ({'type': 'ineq','fun': lambda a: a },
               {'type': 'ineq', 'args': (C,),'fun': lambda a,C: C-a},
               {'type': 'eq', 'args': (y,),'fun': lambda a,y: (a*y).sum()}
              )
    optim = minimize(objective, 
                    x0 = a, 
                    args = ((C,x_train,y_train,identity_kernel,)),
                    constraints=constraints)
    optims.append(optim)
    with open(f'C_{C:.2f}.pkl','wb') as f:
        pickle.dump(optim, f, protocol=pickle.HIGHEST_PROTOCOL)



# In[206]:


from glob import glob
optims = []
for optim_file in glob('./C_*'):
    print(optim_file)
    with open(optim_file, 'rb') as f:
        optims.append(pickle.load(f))
    


# In[207]:


optims[0]


# In[226]:


for i,optim in enumerate(optims): 
    a = optim['x']
    w = np.sum((x_train.T * (a * y_train)),axis=1)
    b = [np.mean(y_train - (x_train @ w))] 
    train_preds = np.sign((x_train @ w) + b)
    test_preds = np.sign((x_test @ w) + b)
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()
    print(f'[w;b] = {np.concatenate((w,b))}')
    print(f'dual SVM w no kernel and C={Cs[i]:2f},\n {train_acc=}, \n {test_acc=}')


# ### Gaussian Kernel w/ different C settings and gamma settings 

# In[183]:


import pickle


# In[ ]:


Cs = [100/873, 500 / 873, 700 / 873]
gammas = [0.1, 0.5, 1, 5, 100]

optims = []
for gamma in gammas:
    for C in Cs:
        y = y_train
        kernel = partial(gaussian_kernel, gamma=gamma)
        a = np.zeros((len(x_train))) 
        constraints = ({'type': 'ineq','fun': lambda a: a },
               {'type': 'ineq', 'args': (C,),'fun': lambda a,C: C-a},
               {'type': 'eq', 'args': (y,),'fun': lambda a,y: (a*y).sum()}
              )
        optim = minimize(objective, 
                        x0 = a, 
                        args = ((C,x_train,y_train,kernel,)),
                        constraints=constraints)
        optims.append(optim)
        with open(f'gaussk_{gamma}_{C:.2f}.pkl','wb') as f:
            pickle.dump(optim, f, protocol=pickle.HIGHEST_PROTOCOL)
            


# In[244]:


from glob import glob
optims_gauss = []
files = glob('./gaussk_*')
files.sort()

for optim_file in files:
    print(optim_file)
    with open(optim_file, 'rb') as f:
        optims_gauss.append(pickle.load(f))


# In[247]:


def kernel_svm_pred(x_train, y_train, x_test, kernel, a):
    preds = np.zeros((len(x_test)))
    b = np.mean(y_train - (x_train @ w))
    for i in range(len(x_test)):
        K = np.array([kernel(x_test[i],ex) for ex in x_train])
        pred = np.sign(((a * y_train * K).sum() + b))
        preds[i] = pred
    return preds


# In[248]:


rep_Cs = Cs*5
for i,optim in enumerate(optims_gauss): 
    file = files[i]
    gamma = float(file.split('_')[1])
    C = rep_Cs[i]
    a = optim['x']
    kernel = partial(gaussian_kernel,gamma=gamma)
    train_preds = kernel_svm_pred(x_train,y_train,x_train,kernel,a)
    test_preds = kernel_svm_pred(x_train,y_train,x_test,kernel,a)
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()
    print(f'[w;b] = {np.concatenate((w,b))}')
    print(f'dual SVM w gauss kernel, {gamma=} and C={C:2f},\n {train_acc=}, \n {test_acc=}')


# In[ ]:


### how many support vectors are the same for different values of gamma? 
### w/ all C = 100 / 873
import itertools
a = np.arange(5)
pairs = list(itertools.product(a, a))
optims_gauss = np.array(optims_gauss)[0,3,6,9,12]
print('begin comparing support vectors for gauss kernel SVM models w/ different gammas')
for pair in pairs: 
    print(f'comparing gamma={gammas[pair[0]]} and gamma={gammas[pair[1]]}')
    support_vecs_1 = optims_gauss[pair[0]]['x'] > 0
    support_vecs_2 = optims_gauss[pair[1]]['x'] > 0
    print(f'num matching support vectors: {np.logical_and(support_vecs_1,support_vecs_2).sum()}')


# ### Bonus Kernel Perceptron Alg

# In[138]:


from functools import partial


# In[140]:


def kernel_perceptron_pred(x_train, y_train, x_test, kernel):
    preds = np.zeros((len(test)))
    for i in range(len(test)):
        K = np.array([kernel(x_test[i],ex) for ex in x_train])
        pred = np.sign((c * y_train * K).sum())
        preds[i] = pred
    return preds


# In[144]:


for gamma in gammas:
    kernel = partial(gaussian_kernel, gamma=gamma)
    print(f'starting gaussian kernel perceptron training w/ {gamma=}')
    c = np.zeros((len(x_train)))
    num_epochs = 10
    train_accs = np.zeros((num_epochs))
    test_accs = np.zeros((num_epochs))
    
    for i in range(num_epochs):
        preds = np.zeros((len(x_train)))
        for j in range(len(x_train)):
            K = np.array([kernel(x_train[j],ex) for ex in x_train])
            pred = np.sign((c[j] * y_train[j] * K).sum())
            if pred != y_train[i]:
                c[j] = c[j] + 1
            preds[j] = pred
        train_accs[i] = (preds == y_train).mean()
        print(f'epoch {i} train_acc = {train_accs[i]}')
    
    test_preds = kernel_perceptron_pred(x_train, y_train, x_test, kernel)
    print(f'test acc: {(test_preds == y_test).mean()}')


# In[81]:





# In[ ]:




