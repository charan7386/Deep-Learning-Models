# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import (load_dataset, random_mini_batches, convert_to_one_hot, 
                      predict)

# ---------------------------- Tensorflow basic Implementations --------------------

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y-y_hat)**2, name='loss')

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
with tf.Session() as sess:
    print(sess.run(c))
sess.close()

sess = tf.Session()
x = tf.placeholder(tf.int64, name='x')
print(sess.run(2*x, feed_dict={x: 3}))
sess.close()

def linear_function():
    
    np.random.seed(1)
    
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)
    
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    
    sess = tf.Session()
    result = sess.run(sigmoid, feed_dict={x: z})
    return result

"""
def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    sess = tf.Session()
    result = sess.run(cost, feed_dict={z:logits, y:labels})
    sess.close()
    return result
"""

def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    result = sess.run(one_hot_matrix)
    sess.close()
    return result

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    result = sess.run(ones)
    sess.close()
    return result

# ---------------------------------- Neural Network with Tensorflow ----------------------------

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#image flattening
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

#normalizing
X_train = X_train_flatten/255
X_test = X_test_flatten/255

#One hot encoding labels
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape=[n_x, None], dtype='float')
    Y = tf.placeholder(shape=[n_y, None], dtype='float')
    return X, Y

def initialize_parameters():
    W1 = tf.get_variable('W1', [25, 12288], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [25, 1], 
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [12, 1], 
                         initializer=tf.zeros_initializer)
    W3 = tf.get_variable('W3', [6, 12], 
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [6, 1], 
                         initializer=tf.zeros_initializer())
    parameters = {'W1': W1,
                 'b1': b1,
                 'W2': W2,
                 'b2': b2,
                 'W3': W3,
                 'b3': b3}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3
    
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, 
          num_epochs=1500, minibatch_size=32, print_cost=True):
    
    # Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax
    
    ops.reset_default_graph()
    
    
    #initialization of model
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    
    #frorward prop
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    
    #backprop
    optimizer = tf.train.GradientDescentOptimizer(learning_rate =
                                                  learning_rate).minimize(cost)

    # Tensorflow variables initializer
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, 
                                                        Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
            if(print_cost and epoch%100==0):
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if(print_cost and epoch%5==0):
                costs.append(epoch_cost)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
    
    
    
    
    
    
    