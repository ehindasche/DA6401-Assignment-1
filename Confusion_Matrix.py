#! DLvenv\Scripts\python.exe

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
from Module_3 import NeuralNetwork

# Initialize wandb
wandb.init(project="fashion-mnist-numpy", name="confusion-matrix")

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocessing
X_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0 # dividing by 255.0 to normalize
X_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0

nn = NeuralNetwork(input_size=784, output_size=10, optimizer='momentum', momentum=0.8, beta=0.999, beta1=0.9, beta2=0.999, init_method='xavier', batch_size=32)
nn.add_layer(512, activation='relu') 
nn.add_layer(512, activation='relu')
nn.add_layer(512, activation="relu")

nn.train(X_train, train_labels, epochs=20, lr=0.01)
test_preds = nn.predict(X_test)
test_accuracy = np.mean(test_preds == test_labels)
print(f'test accuracy: {test_accuracy}')
# Create Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=False, fmt='d', cmap="Reds", linewidths=0.5)

# Highlight the diagonal (correct predictions) in green
for i in range(len(cm)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, edgecolor='green', lw=3))

plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.title("Confusion matrix")

# Save the plot
plt.savefig("confusion_matrix.png")
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

plt.show()
