#! DLvenv\Scripts\python.exe

import numpy as np
from keras.datasets import mnist
import wandb
from NeuralNetwork import NeuralNetwork

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalizing pixel values
# X_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
# X_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

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

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0 # reshaping the 28 x 28 pixel data to a flattened vector of length 784
X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, ratio=0.1)

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'hidden_layers': {'values': [2, 3]},
        'hidden_size': {'values': [512, 256, 128, 64]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-numpy")

def train():
    wandb.init()
    config = wandb.config
    sweep_name = f"Qn10_ls_{config.hidden_layers}_opt_{config.hidden_size}_lr_{config.learning_rate}"
    wandb.run.name = sweep_name
    nn = NeuralNetwork(input_size=784, output_size=10, optimizer='adam')
    for _ in range(config.hidden_layers):
        nn.add_layer(config.hidden_size, activation='relu')
    
    nn.train(X_train, y_train, X_val, y_val, epochs=10, lr=config.learning_rate, loss_type='cross_entropy')
    
wandb.agent(sweep_id, train, count=10)