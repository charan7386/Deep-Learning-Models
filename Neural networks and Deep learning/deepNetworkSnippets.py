# -*- coding: utf-8 -*-

import numpy as np

def initilalize_parameters_deep(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros(layers_dims[l], 1)
    return parameters

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache =  linear_forward(A_prev, W, b)
    if(activation == 'sigmoid'):
        A, activation_cache = sigmoid(Z) #retutns activated value as A and Z as cache
    elif(activation == 'relu'):
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)/2 #parameters = [w1, b1, w2, b2]
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    A_prev = A
    AL, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "sigmoid") #AL activations of final layer
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y, np.log(AL)), np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if(activation == 'sigmoid'):
        dZ = sigmoid_backward(dA, activation_cache) #returns dZ = dA[l]*sigmoid`(Z[l]) 
    elif(activation == 'relu'):
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    current_cache = caches[-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+str(l+1)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu')
    return grads
        
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)/2
    for l in range(1, L+1):
        parameters['W'+str(l)] -= learning_rate*grads['W'+str(l)]
        parameters['b'+str(l)] -= learning_rate*grads['b'+str(l)]
    return parameters