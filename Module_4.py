#! DLvenv\Scripts\python.exe

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb

class NeuralNetwork:
    def __init__(self, input_size, output_size, optimizer='sgd', beta=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = []
        self.activations = []
        self.params = {}
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.velocity = {}
        self.squared_gradients = {}
        self.t = 0

    def add_layer(self, n_units, activation='relu'):
        self.layers.append(n_units) # append change to extend for ease of working
        self.activations.append(activation)

    def initialize_parameters(self):
        layer_dims = [self.input_size] + self.layers + [self.output_size]
        for l in range(1, len(layer_dims)):
            activation_idx = min(l-1, len(self.activations)-1)
            factor = 2. if self.activations[activation_idx] == 'relu' else 1.
            self.params[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(factor / layer_dims[l-1])
            self.params[f'b{l}'] = np.zeros((1, layer_dims[l]))
            
            self.velocity[f'W{l}'] = np.zeros_like(self.params[f'W{l}'])
            self.velocity[f'b{l}'] = np.zeros_like(self.params[f'b{l}'])
            self.squared_gradients[f'W{l}'] = np.zeros_like(self.params[f'W{l}'])
            self.squared_gradients[f'b{l}'] = np.zeros_like(self.params[f'b{l}'])

    @staticmethod
    def one_hot_encode(labels, num_classes):
        labels = np.array(labels).astype(int)
        encoded = np.zeros((len(labels), num_classes))
        encoded[np.arange(len(labels)), labels] = 1
        return encoded
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / expZ.sum(axis=1, keepdims=True)
    
    def forward(self, X, lookahead=False):
        A = X
        self.cache = {'A0': A}
        
        for l in range(1, len(self.layers)+2):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            
            if lookahead and self.optimizer == 'nesterov':
                W -= self.beta * self.velocity[f'W{l}']
                b -= self.beta * self.velocity[f'b{l}']
                
            Z = np.dot(A, W) + b
            
            if l == len(self.layers)+1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z) if self.activations[l-1] == 'relu' else Z
            
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        return A
    
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    
    def backward(self, X, Y, lr=0.01):
        m = Y.shape[0]
        gradients = {}
        L = len(self.layers) + 1
        self.t += 1
        
        dZ = self.cache[f'A{L}'] - Y
        gradients[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ) / m
        gradients[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        for l in reversed(range(1, L)):
            dA = np.dot(dZ, self.params[f'W{l+1}'].T)
            dZ = dA * (self.cache[f'Z{l}'] > 0).astype(float)
            gradients[f'dW{l}'] = np.dot(self.cache[f'A{l-1}'].T, dZ) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        for l in range(1, L+1):
            if self.optimizer in ['momentum', 'nesterov']:
                self.velocity[f'W{l}'] = self.beta * self.velocity[f'W{l}'] + lr * gradients[f'dW{l}']
                self.velocity[f'b{l}'] = self.beta * self.velocity[f'b{l}'] + lr * gradients[f'db{l}']
                self.params[f'W{l}'] -= self.velocity[f'W{l}']
                self.params[f'b{l}'] -= self.velocity[f'b{l}']
            elif self.optimizer == 'rmsprop':
                self.velocity[f'W{l}'] = self.beta * self.velocity[f'W{l}'] + (1-self.beta) * (gradients[f'dW{l}'] **2)
                self.velocity[f'b{l}'] = self.beta * self.velocity[f'b{l}'] + (1-self.beta) * (gradients[f'db{l}'] **2)
                self.params[f'W{l}'] -= lr * gradients[f'dW{l}'] / (np.sqrt(self.velocity[f'W{l}']) + self.epsilon)
                self.params[f'b{l}'] -= lr * gradients[f'db{l}'] / (np.sqrt(self.velocity[f'b{l}']) + self.epsilon)
            elif self.optimizer in ['adam', 'nadam']:
                self.velocity[f'W{l}'] = self.beta * self.velocity[f'W{l}'] + (1 - self.beta) * gradients[f'dW{l}']
                self.velocity[f'b{l}'] = self.beta * self.velocity[f'b{l}'] + (1 - self.beta) * gradients[f'db{l}']
                self.squared_gradients[f'W{l}'] = self.beta2 * self.squared_gradients[f'W{l}'] + (1 - self.beta2) * (gradients[f'dW{l}'] ** 2)
                self.squared_gradients[f'b{l}'] = self.beta2 * self.squared_gradients[f'b{l}'] + (1 - self.beta2) * (gradients[f'db{l}'] ** 2)
                v_w_corrected = self.velocity[f'W{l}'] / (1 - self.beta ** self.t)
                v_b_corrected = self.velocity[f'b{l}'] / (1 - self.beta ** self.t)
                s_w_corrected = self.squared_gradients[f'W{l}'] / (1 - self.beta2 ** self.t)
                s_b_corrected = self.squared_gradients[f'b{l}'] / (1 - self.beta2 ** self.t)
                if self.optimizer == 'nadam':
                    v_w_corrected = self.beta * v_w_corrected + (1 - self.beta) * gradients[f'dW{l}'] / (1 - self.beta2 ** self.t)
                    v_b_corrected = self.beta * v_b_corrected + (1 - self.beta) * gradients[f'db{l}'] / (1 - self.beta2 ** self.t)
                self.params[f'W{l}'] -= lr * v_w_corrected / (np.sqrt(s_w_corrected) + self.epsilon)
                self.params[f'b{l}'] -= lr * v_b_corrected / (np.sqrt(s_b_corrected) + self.epsilon)
            else:
                self.params[f'W{l}'] -= lr * gradients[f'dW{l}']
                self.params[f'b{l}'] -= lr * gradients[f'db{l}']

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000, lr=0.01):
        self.initialize_parameters()
        Y_train_onehot = self.one_hot_encode(Y_train, self.output_size)
        Y_val_onehot = self.one_hot_encode(Y_val, self.output_size)
        
        for epoch in range(epochs):
            Y_hat = self.forward(X_train)
            loss = self.compute_loss(Y_train_onehot, Y_hat)
            self.backward(X_train, Y_train_onehot, lr)
            
            val_preds = self.forward(X_val)
            val_loss = self.compute_loss(Y_val_onehot, val_preds)
            val_accuracy = np.mean(np.argmax(val_preds, axis=1) == Y_val)
            train_accuracy = np.mean(np.argmax(Y_hat, axis=1) == Y_train)
            
            wandb.log({'epoch': epoch+1, 'train_loss': loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy})
            
            print(f"Epoch {epoch+1}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, train_accuracy: {train_accuracy:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Create and train network
# nn = NeuralNetwork(input_size=784, output_size=10, optimizer='adam', beta=0.9, beta2=0.999)
# nn.add_layer(512, activation='relu') # reduced number of neurons in each layer by 1/2 (as a rule of thumb) for high accuracy
# nn.add_layer(256, activation='relu')
# nn.add_layer(128, activation="relu")
# nn.add_layer(64, activation="relu")


# nn.train(X_train, train_labels, epochs=25, lr=0.001)

# # Evaluate
# test_preds = nn.predict(X_test)
# accuracy = np.mean(test_preds == test_labels) # Scope for calculating F1-Score as well to get into better analytics
# # print(len(X_train))
# print(f"Test Accuracy: {accuracy:.4f}")

# Custom train-test split function
def train_test_split(X, Y, ratio=0.1):
    num_samples = X.shape[0]
    num_test = int(num_samples * ratio)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocessing
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0 # reshaping the 28 x 28 pixel data to a flattened vector of length 784
X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, ratio=0.1)

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [30, 40, 50]},
        # 'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [[512, 256, 128], [512,256], [512, 128], [512, 256, 128, 64, 32], [256, 128, 64], [512, 256, 128, 64], [256, 128, 64], [128, 64]]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'adam', 'rmsprop', 'nadam', 'momentum', 'nesterov']},
        # 'batch_size': {'values': [16, 32, 64]},
        'activation': {'values': ['relu']},
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-numpy")

def train():
    wandb.init()
    config = wandb.config
    sweep_name = f"ep_{config.epochs}_ac_{config.activation}_ls_{config.hidden_size}_opt_{config.optimizer}_lr_{config.learning_rate}"
    wandb.run.name = sweep_name
    nn = NeuralNetwork(input_size=784, output_size=10, optimizer=config.optimizer)
    for size in config.hidden_size:
        nn.add_layer(size, activation='relu')
    
    nn.train(X_train, y_train, X_val, y_val, epochs=config.epochs, lr=config.learning_rate)
    
wandb.agent(sweep_id, train, count=20)