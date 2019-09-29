import numpy as np
from tensorflow.keras.datasets import mnist
from numpy import argmax
import matplotlib.pyplot as plt
from keras.utils import to_categorical
(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()
#print("X_train_orig SHAPE = " + str(X_train_orig.shape))
#print("Y_train SHAPE = " + str(Y_train.shape))
#print("X_test_orig SHAPE = " + str(X_test_orig.shape))
#print("Y_test SHAPE = " + str(Y_test.shape))
Y_tr_resh = Y_train_orig.reshape(60000,1)
Y_te_resh = Y_test_orig.reshape(10000,1)
Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
Y_te_T = to_categorical(Y_te_resh, num_classes=10)
Y_train = Y_tr_T.T
Y_test = Y_te_T.T

# Number of Training, Testing and Number of Features
m_train = X_train_orig.shape[0]
m_test = X_test_orig.shape[0]
numpx = X_train_orig.shape[1]

# Flattening of images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
#print("X_train_flatten SHAPE = " + str(X_train_flatten.shape))
#print("X_test_flatten SHAPE = " + str(X_test_flatten.shape))

# Pre-processing
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.


# Activation function
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    return np.maximum(0, x)


# Getting the weights and the biases
def initialize_parameters(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) * 0.01
    return parameters


parameters = initialize_parameters([784, 15, 15, 15, 10])
W1 = parameters["W1"]
W2 = parameters["W2"]
W3 = parameters["W3"]
W4 = parameters["W4"]
b1 = parameters["b1"]
b2 = parameters["b2"]
b3 = parameters["b3"]
b4 = parameters["b4"]

# Defining the functions
Z1 = np.dot(W1, X_train) + b1
Z1_new = []
for i in range(Z1.shape[1]):
    Z1_new = 1 if Z1[0, i] > 0 else 0
A1 = relu(Z1)
Z2 = np.dot(W2, A1) + b2
Z2_new = []
for i in range(Z2.shape[1]):
    Z2_new = 1 if Z2[0, i] > 0 else 0
A2 = relu(Z2)
Z3 = np.dot(W3, A2) + b3
Z3_new = []
for i in range(Z3.shape[1]):
    Z3_new = 1 if Z3[0, i] > 0 else 0
A3 = relu(Z3)
Z4 = np.dot(W4, A3) + b4
A4_out = sigmoid(Z4)
A4 = np.round(A4_out)
# Cost Function
m = Y_train.shape[1]
cost = - np.sum(np.multiply(Y_train, np.log(A4_out)) + (np.multiply((1 - Y_train), (np.log(1 - A4_out))))) / m
#print("Costs: " + str(cost))

# Backprop
num_iterations = 1000
learning_rate = 0.005
costs = []
print_cost = True
for n in range(num_iterations):
    dZ4 = (A4 - Y_train) / m
    dW4 = np.dot(dZ4, A3.T)
    db4 = np.sum(dZ4, axis=1, keepdims=True)
    dA3 = np.dot(W4.T, dZ4)
    dZ3 = np.multiply(dA3, Z3_new)
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, Z2_new)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, Z1_new)
    dW1 = np.dot(dZ1, X_train.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    W1 = W1 - (learning_rate * dW1)
    W2 = W2 - (learning_rate * dW2)
    W3 = W3 - (learning_rate * dW3)
    W4 = W4 - (learning_rate * dW4)
    b1 = b1 - (learning_rate * db1)
    b2 = b2 - (learning_rate * db2)
    b3 = b3 - (learning_rate * db3)
    b4 = b4 - (learning_rate * db4)


    Z1_p = np.dot(W1, X_train) + b1
    Z1_newp = []
    for i in range(Z1_p.shape[1]):
        Z1_newp = 1 if Z1_p[0, i] > 0 else 0
    A1_p = relu(Z1_p)
    Z2_p = np.dot(W2, A1_p) + b2
    Z2_newp = []
    for i in range(Z2_p.shape[1]):
        Z2_newp = 1 if Z2_p[0, i] > 0 else 0
    A2_p = relu(Z2_p)
    Z3_p = np.dot(W3, A2_p) + b3
    Z3_newp = []
    for i in range(Z3_p.shape[1]):
        Z3_newp = 1 if Z3_p[0, i] > 0 else 0
    A3_p = relu(Z3_p)
    Z4_p = np.dot(W4, A3_p) + b4
    A4_outp = sigmoid(Z4_p)
    A4_p = np.round(A4_outp)
    # Cost Function
    m = Y_train.shape[1]
    cost_fn = - np.sum(np.multiply(Y_train, np.log(A4_outp)) + (np.multiply((1 - Y_train), (np.log(1 - A4_outp))))) / m

    if n % 50 == 0:
        costs.append(cost_fn)
    if print_cost and n % 50 == 0:
        print("Cost after iteration %i: %f" % (n, cost_fn))

# Plotting
plt.plot(costs)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()

# Accuracy
print("train accuracy: {} %".format(100 - np.mean(np.abs(A4 - Y_train)) * 100))




