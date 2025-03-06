#! DLvenv\Scripts\python.exe

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist

class NeuralNetwork:
    # Defines the functions required to execute the NeuralNetwork class
    def __init__(self, input_size, output_size):
        self.layers = [] # Creates a list where number of neurons in each layers are stored in sequential order
        self.activations = [] # Stores respective activation functions in sequential order
        self.params = {} # Creates a dictionary where the weights and biases in each layer are stored
        self.input_size = input_size # stores in the input size (dimension of X, 784 in our case)
        self.output_size = output_size # stores the output size (no. of classes, i.e k=10 in our case)

    # Defines a function to add a layer to NN 
    def add_layer(self, n_units, activation='relu'):
        
        self.layers.append(n_units) # adds number of neurons to the list of layers (sequence is from input to output)
        self.activations.append(activation) # adds to activations list

    # Function for parameter initialization (random initialization) 
    def initialize_parameters(self):
        layer_dims = [self.input_size] + self.layers + [self.output_size] # layer_dims is an array with number of neurons stacked in sequential order (input to output)
        for l in range(1, len(layer_dims)): # range is from 1 as no weights before input layer (first hidden layer is layer 1 and output layer is layer L)
            # Using min() to prevent index overflow for output layer
            activation_idx = min(l-1, len(self.activations)-1)
            factor = 2. if self.activations[activation_idx] == 'relu' else 1. 
            self.params[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(factor / layer_dims[l-1])
            self.params[f'b{l}'] = np.zeros((1, layer_dims[l]))
            # the notation for shape of weights and biases has been transposed as opposed to one discussed in class
            # this is based on my convenience of handling dimensions across all functions
            
    @staticmethod
    def one_hot_encode(labels, num_classes):
        labels = np.array(labels).astype(int)
        encoded = np.zeros((len(labels), num_classes)) # defines a matrix with dimensions of (Y[0], 10)
        encoded[np.arange(len(labels)), labels] = 1 # puts a 1 wherever the index corresponds to the label
        return encoded
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # np.max(Z, axis=1) subtracted for stability of exponential term. It still preserves the relative differences between Zi.
        return expZ / expZ.sum(axis=1, keepdims=True) # axis = 1 as neurons spread along columns and datapoints along rows
    
    def forward(self, X):
        A = X
        self.cache = {'A0': A} # input layer is A0 = X
        
        for l in range(1, len(self.layers)+2):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            Z = np.dot(A, W) + b # pre-activation (produces shape of n x dim(l), where n = 60000 for X_train)
            
            if l == len(self.layers)+1:  # Output layer
                A = self.softmax(Z) # applies softmax for output layer pre-activation
            else:
                A = self.relu(Z) if self.activations[l-1] == 'relu' else Z # for output layer, we use softmax as above
            
            self.cache[f'Z{l}'] = Z # pre-activation of layer l stored to cache
            self.cache[f'A{l}'] = A # post-activation of layer l stored to cache
            
        return A # returns prediction of ouput layer
    
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m # we use cross entropy, divided by total number of samples to prevent shooting of value
        # the 1e-8 is used to avoid the log term from exploding if Y_hat == 0
    
    def backward(self, X, Y, lr=0.01):
        m = Y.shape[0] # m = no. of datapoints, 60000 for X_test
        gradients = {} # stores gradients of weights and biases at every layer
        L = len(self.layers) + 1  # Total layers (self.layers only considers hidden layers. +1 ensures inclusion of output layer)
        
        # Output layer gradient
        dZ = self.cache[f'A{L}'] - Y
        # print(f'shape of dZ {dZ.shape}')
        gradients[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ) / m # gradient update rule w.r.t parameters as discussed in the lectures
        # print(f"shape of dW{L}: {gradients[f'dW{L}'].shape}") 
        gradients[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        # print(f"shape of db{L}: {gradients[f'db{L}'].shape}")
        
        # Hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.params[f'W{l+1}'].T) # gradient w.r.t to post-activation
            # print(f'shape of dA {dA.shape}')
            dZ = dA * (self.cache[f'Z{l}'] > 0).astype(float) # gradient w.r.t to pre-activation
            gradients[f'dW{l}'] = np.dot(self.cache[f'A{l-1}'].T, dZ) / m # gradient of weights of layer l
            # print(f"shape of dW{l}: {gradients[f'dW{l}'].shape}")
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m # gradient of biases of layer l
            # print(f"shape of db{l}: {gradients[f'db{l}'].shape}")
        # Update parameters using simple gradient descent
        for l in range(1, L+1):
            self.params[f'W{l}'] -= lr * gradients[f'dW{l}']
            self.params[f'b{l}'] -= lr * gradients[f'db{l}']
            
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1) # argmax returns index of the one with highest probability value
    
    def train(self, X_train, Y_train, epochs=1000, lr=0.01): 
        self.initialize_parameters()
        Y_train_onehot = self.one_hot_encode(Y_train, self.output_size)
        
        for epoch in range(epochs):
            Y_hat = self.forward(X_train) 
            loss = self.compute_loss(Y_train_onehot, Y_hat)
            self.backward(X_train, Y_train_onehot, lr)
            
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocessing
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0 # reshaping the 28 x 28 pixel data to a flattened vector of length 784

# Create and train network
nn = NeuralNetwork(input_size=784, output_size=10)
nn.add_layer(392, activation='relu') # reduced number of neurons in each layer by 1/2 (as a rule of thumb) for increasing model accuracy
nn.add_layer(196, activation='relu')
nn.add_layer(98, activation="relu")


nn.train(X_train, train_labels, epochs=10, lr=0.1)

# Evaluate
test_preds = nn.predict(X_test)
accuracy = np.mean(test_preds == test_labels) # Scope for calculating F1-Score as well to get into better analytics
# print(len(X_train))
print(f"Test Accuracy: {accuracy:.4f}")
