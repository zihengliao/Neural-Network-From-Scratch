from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot
import numpy as np



np.random.seed(0)


 #inputs
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# im gonna have 16 neurons. not sure why. saw 3blue1brown do it 

class Layer:
    def __init__(self, n_inputs, n_neurons) -> None:
        # self.weights = np.arange(n_neurons*28).reshape(28,n_neurons)        # number 28 because inputs are 28x28
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


# using the relu function as it requires less computing power
class Relu:
    def forward(self, neurons):
        return np.maximum(0, neurons)

class Softmax:
    def forward(self, inputs):
        # print(np.max(inputs, axis = 1, keepdims= True).shape)
        exps = np.exp(inputs.T - np.max(inputs.T, axis = 1, keepdims= True))
        # print(exps.shape)
        probabilities = exps / np.sum(exps, axis = 1, keepdims= True)
        return probabilities

# first_layer = np.arange(16*28).reshape(28,16)
# print(first_layer.shape)
# print(train_X[0].shape)

first_layer = Layer(28, 16)
second_layer = Layer(16, 10)
output_layer = Layer(28,1)
activation = Relu()
activation_soft = Softmax()

first_layer.forward(np.array(train_X[0]))
first_layer_output = activation.forward(first_layer.output)

second_layer.forward(first_layer_output)
second_layer_output = activation.forward(second_layer.output).T
print(second_layer_output.shape)
output_layer.forward(second_layer_output)
# print(output_layer.output.shape)
result = activation_soft.forward(output_layer.output)


print(result.shape)
# print()
# print()
# print(result.shape)
# print()
# print()
# print(second_layer.output)
