#! DLvenv\Scripts\python.exe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

# Initialize a W&B run
wandb.init(project="fashion-mnist-numpy", name="sample-images")

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create a W&B Table for logging
table = wandb.Table(columns=["Image", "Label"])

# Select and log one image per class
for i in range(10):
    idx = np.where(y_train == i)[0][0]  # Get first occurrence of each class
    img = x_train[idx]  # Get image
    label = class_labels[i]  # Get label
    
    # Add image and label to W&B Table
    table.add_data(wandb.Image(img), label)

# Log the table directly (this makes it visible in the UI)
wandb.log({"Fashion-MNIST Samples": table})

# Create an artifact to store the table
artifact = wandb.Artifact(name="fashion_mnist_numpy", type="dataset")
artifact.add(table, "samples")  # Fix: Add table properly to artifact

# Log the artifact
wandb.log_artifact(artifact)

# Finish the run
wandb.finish()