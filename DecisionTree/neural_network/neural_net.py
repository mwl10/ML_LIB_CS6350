from jax import numpy as jnp
from jax import random


def init_params(random_key, n_x, n_h, n_y):
    # n_x -- num input features
    # n_h -- size of the 2 hidden layers
    # n_y -- size of the output layer
    keys = random.split(random_key, 3)
    return {
            "W0": random.uniform(keys[0], shape=(n_h,n_x)),
            "b0": jnp.zeros((n_h,)),
            "W1": random.uniform(keys[1], shape=(n_h,n_h)),
            "b1": jnp.zeros((n_h,)),
            "W2": random.uniform(keys[2], shape=(n_y, n_h)),
            "b2": jnp.zeros((n_y,))
            }

def sigmoid(Z):
    A = 1/(1+jnp.exp(-Z))
    cache = Z ## keep for backprop 
    return A, cache

def sigmoid_backprop(dL_dA, cache):
    Z = cache
    s = 1/(1+jnp.exp(-Z))
    dL_dZ = dL_dA * s * (1-s)
    return dL_dZ


def layer_forward(A, W, b):
    Z = A @ W.T + b
    linear_cache = (A, W, b) #### keep for backprop
    A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def model_forward(X, parameters):
    A0,cache0 = layer_forward(X, 
                              parameters['W0'], 
                              parameters['b0'])
    A1,cache1 = layer_forward(A0, 
                              parameters['W1'], 
                              parameters['b1'])
    A2, cache2 = layer_forward(A1, 
                              parameters['W2'], 
                              parameters['b2'])
    caches = (cache0,cache1,cache2)
    return A2, caches

def layer_backprop(dL_dA, cache):
    linear_cache, activation_cache = cache
    dL_dZ = sigmoid_backprop(dL_dA, activation_cache)
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dL_dW = (dL_dZ.T @ A_prev) / m
    dL_db = dL_dZ.sum(axis=0, keepdims=True) / m
    dL_dA_prev = dL_dZ @ W
    return dL_dA_prev, dL_dW, dL_db

def model_backprop(preds, y, caches):
    grads = {}
    L = 3

    dL_dA2 = preds - y
    dL_dA1, dL_dW2, dL_db2 = layer_backprop(dL_dA2, caches[L-1])
    grads["dL_dA1"] = dL_dA1
    grads["dL_dW2"] = dL_dW2
    grads["dL_db2"] = dL_db2

    dL_dA0, dL_dW1, dL_db1 = layer_backprop(dL_dA1, caches[L-2])
    grads["dL_dA0"] = dL_dA0
    grads["dL_dW1"] = dL_dW1
    grads["dL_db1"] = dL_db1

    _, dL_dW0, dL_db0 = layer_backprop(dL_dA0, caches[L-3])
    grads["dL_dW0"] = dL_dW0
    grads["dL_db0"] = dL_db0
    return grads

def update_params(params, grads, lr):
    p = params.copy()
    p['W0'] = p['W0'] - lr * grads['dL_dW0']
    p['b0'] = p['b0'] - lr * grads['dL_db0']
    p['W1'] = p['W1'] - lr * grads['dL_dW1']
    p['b1'] = p['b1'] - lr * grads['dL_db1']     
    p['W2'] = p['W2'] - lr * grads['dL_dW2']
    p['b2'] = p['b2'] - lr * grads['dL_db2']                                                    
                                                  
    return p