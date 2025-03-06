#! DLvenv\Scripts\python.exe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Loading the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# ABOVE DATASET HAS SHAPES: x_train - (60000,28,28), y_train - (60000,), x_test - (10000,28,28), y_test - (10000,)
# (28,28) refers to pixel values ranging from 0 to 255, where 0 represents total black and 255 represents total white color of the pixel

# Class labels for Fashion-MNIST
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Extracting one sample for each class
sample_images = []
sample_labels = []
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    # np.where() returns an array with each row containing indices of occurence of that label (0th row for i==0)
    # We thus select first index of this every row (idx)
    sample_images.append(x_train[idx])
    sample_labels.append(class_labels[i])

# Plotting images in a grid
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.set_title(sample_labels[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

