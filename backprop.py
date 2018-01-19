# -*- coding: utf-8 -*-
"""
Numpy translation of this article:
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
with some generalization to classes/reusable functions.

This script is not intended for use, just to get a better grasp of backprop mechanics.
"""
import numpy
from matplotlib import pyplot

# Store a layer with an activation function, the inverse derivative
# function, and arrays for inputs, values, and deltas
class Layer:
    def __init__(self, size, fn, fn_deriv):
        self.size = size
        self.fn = fn
        self.fn_deriv = fn_deriv
        self.inputs = numpy.zeros((size,1))
        self.values = numpy.zeros((size,1))
        self.deltas = numpy.zeros((size,1))
    
    def __str__(self):
        return 'Layer: ' + self.values.__str__()
    
    # Apply the activation function to layer inputs
    def set_inputs(self, inputs):
        self.inputs = inputs
        self.values = self.fn(self.inputs)
    
    # Apply the derivative to layer outputs
    def set_deltas(self, errors):
        self.deltas = errors * self.fn_deriv(self.values)


# Initialize a weight matrix to connect two layers using mean 0 and std 1
def rand_weights(layer_in, layer_out):
    return numpy.random.randn(layer_in.size + 1, layer_out.size)


# Store a list of layers connected by weight matrices which specifies
# a simple backprop network
class Network:
    def __init__(self):
        self.layers = []
        self.weights = []
    
    # Add a layer to the network of the specified size using the activation
    # function and derivative given
    def add_layer(self, size, fn, fn_deriv):
        curr_layer = Layer(size, fn, fn_deriv)
        if (len(self.layers) > 0):
            prev_layer = self.layers[-1]
            self.weights.append(rand_weights(prev_layer, curr_layer))
        self.layers.append(curr_layer)
    
    # Pass the inputs forward through the network, setting node values
    def forward(self, inputs):
        prev_layer = self.layers[0]
        prev_layer.set_inputs(inputs)
        for curr_layer, weights in zip(self.layers[1:], self.weights):
            inputs = numpy.hstack((prev_layer.values, [1])) @ weights
            curr_layer.set_inputs(inputs)
            prev_layer = curr_layer
    
    # Pass the targets backward through the network, setting node deltas, return the SSE
    def backward(self, targets):
        output_layer = self.layers[-1]
        output_errors = targets - output_layer.values
        output_layer.set_deltas(output_errors)
        curr_errors = output_errors
        for i in range(len(self.layers) - 2, -1, -1):
            prev_errors = curr_errors @ self.weights[i].transpose()
            prev_errors = prev_errors[:-1]
            self.layers[i].set_deltas(prev_errors)
            curr_errors = self.layers[i].deltas
        return (output_errors ** 2).mean()
    
    # Train the network given the current node deltas and weight matrices
    def train(self, alpha):
        for prev_layer, curr_layer, weights in zip(self.layers[:-1], self.layers[1:], self.weights):
            weights += alpha * (numpy.hstack((prev_layer.values, [1])).reshape((prev_layer.size + 1,1)) @ curr_layer.deltas.reshape((1,curr_layer.size)))
    
    # Test the network on the inputs given the targets, return the SSE
    def test(self, inputs, targets):
        self.forward(inputs)
        return ((targets - self.layers[-1].values) ** 2).mean()            


# Linear activation function does not transform inputs
def linear(x):
    return x


# Linear derivative is 1 everywhere
def linear_deriv(y):
    return numpy.ones_like(y)


# Logistic activation function squashes the high and low tails to (0,1)
def logistic(x):
    return 1 / (1 + numpy.exp(-x))


def logistic_deriv(y):
    return y * (1 - y)


# Rectified linear unit activation with a non-zero low cutoff, to preserve gradients
def relu(x):
    return numpy.maximum(x, 0.01 * x)


# If x is negative, y is negative, so the derivative is the smaller slope
def relu_deriv(y):
    return ((y < 0) * 0.1) + ((y > 0) * 1)


# Create a 4 layer network using logistic
numpy.random.seed(31)
net = Network()
net.add_layer(2, linear, linear_deriv)
net.add_layer(7, logistic, logistic_deriv)
net.add_layer(1, logistic, logistic_deriv)

# Setup an XOR training dataset
x = numpy.array([[0,0], [1,0], [0,1], [1,1]])
y = numpy.array([[0.0], [1.0], [1.0], [0.0]])

# Train the network for 5000 epochs with minibatch size = 1
for epoch in range(5000):
    sse = 0
    for i in range(4):
        net.forward(x[i,:])
        sse += net.backward(y[i,:])
        net.train(0.1)
    if (epoch % 250 == 0):
        print('SSE: {}'.format(sse))

# Print the network structure
for layer, weights in zip(net.layers[:-1], net.weights):
    print('Weights:')
    print(weights)
    print()

# Print the network inputs and outputs for each training row
for i in range(x.shape[0]):
    net.forward(x[i,:])
    print(net.layers[0])
    print(net.layers[-1])


# Setup a dataset consisting of two linked spirals in a 2d plane
zeros_theta = numpy.linspace(0, 2 * numpy.pi, 1000)
zeros_r = numpy.linspace(1, 3, 1000)
zeros_x = numpy.vstack((zeros_r * numpy.cos(zeros_theta),
                        zeros_r * numpy.sin(zeros_theta))).transpose()
ones_theta = numpy.linspace(numpy.pi, 3 * numpy.pi, 1000)
ones_r = numpy.linspace(1, 3, 1000)
ones_x = numpy.vstack((ones_r * numpy.cos(ones_theta),
                       ones_r * numpy.sin(ones_theta))).transpose()

# Combine and randomize the classes and labels
x = numpy.vstack((zeros_x, ones_x))
y = numpy.hstack((numpy.zeros(1000), numpy.ones(1000))).reshape((2000,1))
order = numpy.random.permutation(2000)
x = x[order]
y = y[order]

# Visualize the dataset
pyplot.scatter(x[:,0], x[:,1], c=y, edgecolor='face')

# Create a larger network
numpy.random.seed(58)
net = Network()
net.add_layer(2, linear, linear_deriv)
net.add_layer(10, relu, relu_deriv)
net.add_layer(10, relu, relu_deriv)
#net.add_layer(50, relu, relu_deriv)
net.add_layer(1, relu, relu_deriv)

# Train the network for 5000 epochs with minibatch size = 1
for epoch in range(1000):
    sse = 0
    for i in range(x.shape[0]):
        net.forward(x[i,:])
        sse += net.backward(y[i,:])
        net.train(0.001)
    if (epoch % 10 == 0):
        print('Epoch: {} SSE: {}'.format(epoch, sse))

# Visualize the network performance
pred = numpy.zeros_like(y)
for i in range(x.shape[0]):
    net.test(x[i,:], y[i,:])
    pred[i] = net.layers[-1].values
pyplot.scatter(x[:,0], x[:,1], c=pred, edgecolor='face')
