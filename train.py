#! DLvenv\Scripts\python.exe

import argparse
import wandb
from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network with specified hyperparameters.")

    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")

    # Dataset arguments
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use for training.")

    # Training hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used for training.")

    # Loss function
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",
                        help="Loss function used for training.")

    # Optimizer arguments
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        default="sgd", help="Optimizer used for training.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate for optimization.")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for 'momentum' and 'nag' optimizers.")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for 'rmsprop' optimizer.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for 'adam' and 'nadam' optimizers.")
    parser.add_argument("--beta2", type=float, default=0.5, help="Beta2 for 'adam' and 'nadam' optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon value for optimizers.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="Weight decay used in optimization.")

    # Network architecture arguments
    parser.add_argument("-wi", "--weight_init", type=str, choices=["random", "Xavier", "he"], default="random",
                        help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons in each hidden layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "relu"],
                        default="sigmoid", help="Activation function.")

    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
    X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    # Initialize and train the model
    nn = NeuralNetwork(
        input_size=784,
        output_size=10,
        optimizer=args.optimizer,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        init_method=args.weight_init,
        batch_size=args.batch_size
    )

    X_train, X_val, y_train, y_val = nn.train_test_split(X_train, train_labels, ratio=0.1)

    for _ in range(args.num_layers):
        nn.add_layer(args.hidden_size, activation=args.activation)
    
    nn.train(X_train, y_train, X_val, y_val, epochs=args.epochs, lr=args.learning_rate, loss_type=args.loss)

if __name__ == '__main__':
    main()
