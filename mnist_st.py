import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
import time

(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()

# Streamlit
st.title('MNIST Digits Classification with Deep Learning using python and Numpy')
st.sidebar.title('Parameters')

# Preparing the data
Y_tr_resh = Y_train_orig.reshape(60000, 1)
Y_te_resh = Y_test_orig.reshape(10000, 1)
Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
Y_te_T = to_categorical(Y_te_resh, num_classes=10)
Y_train = Y_tr_T.T
Y_test = Y_te_T.T


# Flattening of inputs
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Plotting some digits
second_image = X_train_orig[1, :]
second_label = Y_tr_resh[1]
#plt.imshow(second_image, cmap='gray_r')
#plt.title('Digit Label: {}'.format(second_label))
#st.pyplot()





# Preprocessing of Inputs
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.



# Defining the activation functions
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(p):
    return np.maximum(0, p)


def softmax(u):
    return np.exp(u) / np.sum(np.exp(u), axis=0, keepdims=True)

# Initializing the Weights and Biases
hidd_lay = st.sidebar.slider("Number of  hidden layers: ", 1, 10, 3, 1)
hidd_units = {}
layer_val = [784]
for l in range(1, hidd_lay+1):
    hidd_units["h" + str(l)] = st.sidebar.slider("Number of units in hidden layer " + str(l), 10, 100, 50, 5)
    layer_val.append(hidd_units["h" + str(l)])
layer_val.append(10)







parameters = {}
def initialize_parameters(layer_dims):
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * (np.sqrt(2 / layer_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
#initialize_parameters([X_train.shape[0], 50, 50, 50, 10])

#for l in range(1, 5):
    #print("W" + str(l) + " = " + str(parameters["W" + str(l)]))
    #print("W" + str(l) + "shape" + " = " + str(parameters["W" + str(l)].shape))
    #print("b" + str(l) + " = " + str(parameters["b" + str(l)]))
    #print("b" + str(l) + "shape" + " = " + str(parameters["b" + str(l)].shape))

# Forward Propagation
outputs = {}
activation = {}
def forward_prop(parameters, X_train, activation):
    L = len(layer_val)
    outputs["Z" + str(1)] = np.dot(parameters["W1"], X_train) + parameters["b1"]
    activation["A" + str(1)] = relu(outputs["Z" + str(1)])
    for l in range(2, L-1):
        outputs["Z" + str(l)] = np.dot(parameters["W" + str(l)], activation["A" + str(l - 1)]) + parameters["b" + str(l)]
        activation["A" + str(l)] = relu(outputs["Z" + str(l)])
    outputs["Z" + str(L-1)] = np.dot(parameters["W" + str(L-1)], activation["A" + str(L-2)]) + parameters["b" + str(L-1)]
    activation["A" + str(L-1)] = softmax(outputs["Z" + str(L-1)])
    return outputs, activation
#forward_prop(parameters, X_train, activation)

# Computing the Cost
def compute_cost(activation):
    loss = - np.sum((Y_train * np.log(activation["A" + str(len(layer_val)-1)])), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / m
    return cost


# Backward Propagation with Regular Gradient Descent
def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
grad_reg = {}
m = X_train.shape[1]
def grad_re(parameters, outputs, activation):
    L = len(layer_val)
    grad_reg["dZ" + str(L-1)] = (activation["A" + str(L-1)] - Y_train) / m
    for l in range(1, L-1):
        grad_reg["dA" + str(L-1 - l)] = np.dot(parameters["W" + str(L-1 - l + 1)].T, grad_reg["dZ" + str(L-1 - l + 1)])
        grad_reg["dZ" + str(L-1 - l)] = grad_reg["dA" + str(L-1 - l)] * drelu(outputs["Z" + str(L-1 - l)])
    grad_reg["dW1"] = np.dot(grad_reg["dZ1"], X_train.T)
    grad_reg["db1"] = np.sum(grad_reg["dZ1"], axis=1, keepdims=True)
    for l in range(2, L):
        grad_reg["dW" + str(l)] = np.dot(grad_reg["dZ" + str(l)], activation["A" + str(l - 1)].T)
        grad_reg["db" + str(l)] = np.sum(grad_reg["dZ" + str(l)], axis=1, keepdims=True)
    return parameters, outputs, activation, grad_reg
#grad_re(parameters, outputs, activation)
def learning(grad_reg, learning_rate):
    for i in range(1, len(layer_val)):
        parameters["W" + str(i)] = parameters["W" + str(i)] - (learning_rate * grad_reg["dW" + str(i)])
        parameters["b" + str(i)] = parameters["b" + str(i)] - (learning_rate * grad_reg["db" + str(i)])
    return parameters
#learning(parameters, grad_reg, learning_rate=0.005)

# Predictions
def predict(parameters, X_test):
    forward_prop(parameters, X_test, activation)
    predictions = np.round(activation["A" + str(len(layer_val)-1)])
    return predictions


# Iterating over num_iterations
num_iterations = st.sidebar.slider("Number of iterations: ", 0, 5000, 1000, 100)
lr_alpha = float(st.sidebar.text_input("Learning Rate: "))
print_cost = True
costs = []

# Gradient Descent over number of iterations
def grad_descent(num_iterations, costs, activation):
    initialize_parameters(layer_val)
    for l in range(0, num_iterations):
        tick = time.time()
        forward_prop(parameters, X_train, activation)
        cost = compute_cost(activation)
        grad_re(parameters, outputs, activation)
        learning(grad_reg, learning_rate=lr_alpha)
        tock = time.time()
        time_taken = (tock - tick)
        if l % 100 == 0:
            costs.append(cost)
        if print_cost and l % 100 == 0:
            print("Cost after iteration %i: %f       time taken: %f" % (l, cost, time_taken))
            st.write("Cost after iteration %i: %f       time taken: %f" % (l, cost, time_taken))
    return costs, parameters

if st.button('Train model'):
    st.write('Training the model...')
    grad_descent(num_iterations, costs, activation)
else:
    st.write(' ')


# Visualising the Cost vs num_iterations
plt.plot(costs)
plt.title('Gradient Descent')
plt.xlabel('iterations')
plt.ylabel('cost')
st.pyplot()
