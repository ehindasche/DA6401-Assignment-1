#! DLvenv\Scripts\python.exe

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb

class NeuralNetwork:
    def __init__(self, input_size, output_size, optimizer='sgd', momentum = 0.9, beta=0.9, beta1 = 0.9, beta2=0.999, epsilon=1e-8, init_method='he', batch_size=32):
        self.layers = []
        self.activations = []
        self.params = {}
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.velocity = {}
        self.squared_gradients = {}
        self.t = 0
        self.init_method = init_method  # Set initialization method
        self.batch_size = batch_size    # Batch size parameter

    def add_layer(self, n_units, activation='relu'):
        self.layers.append(n_units)
        self.activations.append(activation)

    def initialize_parameters(self):
        layer_dims = [self.input_size] + self.layers + [self.output_size]

        for l in range(1, len(layer_dims)):
            n_in, n_out = layer_dims[l-1], layer_dims[l]
            activation_idx = min(l-1, len(self.activations)-1)
            activation = self.activations[activation_idx]

            if self.init_method == 'xavier':
                # Xavier (Glorot) initialization
                limit = np.sqrt(6 / (n_in + n_out))
                self.params[f'W{l}'] = np.random.uniform(-limit, limit, (n_in, n_out))

            elif self.init_method == 'he':
                # He initialization (for ReLU)
                self.params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

            else:
                # Random small values (default)
                self.params[f'W{l}'] = np.random.randn(n_in, n_out) * 0.01

            self.params[f'b{l}'] = np.zeros((1, n_out))

            # Initialize momentum and RMSprop terms
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
                W -= self.momentum * self.velocity[f'W{l}']
                b -= self.momentum * self.velocity[f'b{l}']
                
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

    # def mean_squared_error(self, Y, Y_hat):
    #     return np.sum((Y - Y_hat)**2)
    
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
                self.velocity[f'W{l}'] = self.momentum * self.velocity[f'W{l}'] + lr * gradients[f'dW{l}']
                self.velocity[f'b{l}'] = self.momentum * self.velocity[f'b{l}'] + lr * gradients[f'db{l}']
                self.params[f'W{l}'] -= self.velocity[f'W{l}']
                self.params[f'b{l}'] -= self.velocity[f'b{l}']
            elif self.optimizer == 'rmsprop':
                self.velocity[f'W{l}'] = self.beta * self.velocity[f'W{l}'] + (1-self.beta) * (gradients[f'dW{l}'] **2)
                self.velocity[f'b{l}'] = self.beta * self.velocity[f'b{l}'] + (1-self.beta) * (gradients[f'db{l}'] **2)
                self.params[f'W{l}'] -= lr * gradients[f'dW{l}'] / (np.sqrt(self.velocity[f'W{l}']) + self.epsilon)
                self.params[f'b{l}'] -= lr * gradients[f'db{l}'] / (np.sqrt(self.velocity[f'b{l}']) + self.epsilon)
            elif self.optimizer in ['adam', 'nadam']:
                self.velocity[f'W{l}'] = self.beta1 * self.velocity[f'W{l}'] + (1 - self.beta1) * gradients[f'dW{l}']
                self.velocity[f'b{l}'] = self.beta1 * self.velocity[f'b{l}'] + (1 - self.beta1) * gradients[f'db{l}']
                self.squared_gradients[f'W{l}'] = self.beta2 * self.squared_gradients[f'W{l}'] + (1 - self.beta2) * (gradients[f'dW{l}'] ** 2)
                self.squared_gradients[f'b{l}'] = self.beta2 * self.squared_gradients[f'b{l}'] + (1 - self.beta2) * (gradients[f'db{l}'] ** 2)
                v_w_corrected = self.velocity[f'W{l}'] / (1 - self.beta1 ** self.t)
                v_b_corrected = self.velocity[f'b{l}'] / (1 - self.beta1 ** self.t)
                s_w_corrected = self.squared_gradients[f'W{l}'] / (1 - self.beta2 ** self.t)
                s_b_corrected = self.squared_gradients[f'b{l}'] / (1 - self.beta2 ** self.t)
                if self.optimizer == 'nadam':
                    v_w_corrected = self.beta1 * v_w_corrected + (1 - self.beta1) * gradients[f'dW{l}'] / (1 - self.beta2 ** self.t)
                    v_b_corrected = self.beta1 * v_b_corrected + (1 - self.beta1) * gradients[f'db{l}'] / (1 - self.beta2 ** self.t)
                self.params[f'W{l}'] -= lr * v_w_corrected / (np.sqrt(s_w_corrected) + self.epsilon)
                self.params[f'b{l}'] -= lr * v_b_corrected / (np.sqrt(s_b_corrected) + self.epsilon)
            else:
                self.params[f'W{l}'] -= lr * gradients[f'dW{l}']
                self.params[f'b{l}'] -= lr * gradients[f'db{l}']

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000, lr=0.01):
        self.initialize_parameters()
        Y_train_onehot = self.one_hot_encode(Y_train, self.output_size)
        Y_val_onehot = self.one_hot_encode(Y_val, self.output_size)

        m = X_train.shape[0]  # Number of training samples

        for epoch in range(epochs):
            indices = np.random.permutation(m)  # Shuffle dataset
            X_train, Y_train_onehot = X_train[indices], Y_train_onehot[indices]

            num_batches = m // self.batch_size

            epoch_loss = 0
            epoch_correct = 0

            for i in range(0, m, self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                Y_batch = Y_train_onehot[i:i + self.batch_size]

                Y_hat = self.forward(X_batch, lookahead=(self.optimizer == 'nesterov'))
                self.backward(X_batch, Y_batch, lr)

                # Compute loss for the batch
                epoch_loss += self.compute_loss(Y_batch, Y_hat) * len(Y_batch)

                # Compute correct predictions
                epoch_correct += np.sum(np.argmax(Y_hat, axis=1) == np.argmax(Y_batch, axis=1))

            # Average loss and accuracy over all mini-batches
            epoch_loss /= m
            train_accuracy = epoch_correct / m

            # Compute validation accuracy and loss in a single batch
            val_preds = self.forward(X_val)
            val_loss = self.compute_loss(Y_val_onehot, val_preds)
            val_accuracy = np.mean(np.argmax(val_preds, axis=1) == Y_val)

            wandb.log({'epoch': epoch+1, 'train_loss': epoch_loss, 'val_loss': val_loss,
                    'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy})

            print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")


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
        'batch_size': {'values':[32, 64]},
        'weight_init': {'values': ['he', 'xavier']},
        'epochs': {'values': [15]},
        'hidden_layers': {'values': [2, 3, 4, 5]},
        'hidden_size': {'values': [512, 256, 128, 64]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'adam', 'rmsprop', 'nadam', 'momentum', 'nesterov']},
        'activation': {'values': ['relu']},
        'momentum': {'values': [0.8, 0.9, 0.99]},
        'beta': {'values': [0.9, 0.999]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-numpy")

def train():
    wandb.init()
    config = wandb.config
    sweep_name = f"ac_{config.activation}_ls_{config.hidden_layers}_opt_{config.optimizer}_lr_{config.learning_rate}"
    wandb.run.name = sweep_name
    nn = NeuralNetwork(input_size=784, output_size=10, optimizer=config.optimizer, init_method= config.weight_init, momentum=config.momentum, beta=config.beta, batch_size=config.batch_size)
    for _ in range(config.hidden_layers):
        nn.add_layer(config.hidden_size, activation='relu')
    
    nn.train(X_train, y_train, X_val, y_val, epochs=config.epochs, lr=config.learning_rate)
    
wandb.agent(sweep_id, train, count=15)