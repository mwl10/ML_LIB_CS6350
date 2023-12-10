from jax import numpy as jnp
from jax import random


def init_params(n_x, n_y):
    # n_x -- num input features
    # n_y -- size of the output layer
    return {
            "W0": jnp.zeros((n_y,n_x)),
            "b0": jnp.zeros((n_y,)),
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
    caches = (cache0,)
    return A0, caches

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
    L = 1
    dL_dA0 =  - ((y / preds) - ((1 - y) / (1 - preds))) ### d(BCE loss)/d(activations)
    _, dL_dW0, dL_db0 = layer_backprop(dL_dA0, caches[0])
    grads["dL_dW0"] = dL_dW0
    grads["dL_db0"] = dL_db0
    return grads

def update_params(params, grads, lr):
    p = params.copy()
    p['W0'] = p['W0'] - lr * grads['dL_dW0']
    p['b0'] = p['b0'] - lr * grads['dL_db0']                                                 
                                          
    return p