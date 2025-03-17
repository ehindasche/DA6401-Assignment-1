# DA6401-Assignment-1
This project involves the development and implementation of neural network architectures from fundamental principles using NumPy. The Fashion-MNIST dataset serves as the training and evaluation corpus. Comprehensive experiment tracking and results visualization are achieved through the integration of Weights & Biases (wandb.ai), facilitating detailed reporting and insightful analysis.

**Report Link:** https://wandb.ai/eshan_kulkarni-indian-institute-of-technology-madras/fashion-mnist-numpy/reports/DA6401-Assignment-1--VmlldzoxMTcwNjQ1NA?accessToken=4ikfk15cx4kg0cn067e7wrnbnxhubaxbduyquy3twomub091dl7jxt802lb7r0jg

## Step 1
Module1.py contains the code for Question 1 of the assignment.
- **requirements: Self sufficient**

## Step 2
NeuralNetwork.py contains code for Question 2-3. It contains code for the entire Neural Network class with necessary functions and optimizers included in the class.
- **requirements: Self sufficient**

## Step 3
wandb.py contains code for sweep functionality. The charts obtained in Question 4-6 and also the ones for "MSE" in question 8 have been obtained through this.
- **requirements: Self sufficient**

## Step 4
confusion_matrix.py generates the confusion matrix. It uses code from the Module_3.py file.
The file was run by training the NeuralNetwork on the best parameters obtained (mentioned in the report)
- **requirements: NeuralNetwork class from Module_3.py**

## Step 5
The mnist.py file uses the sweep functionality to run sweeps based on only 3 parameters. This is mentioned in the report along with the charts attached. 
- **requirements: NeuralNetwork class from NeuralNetwork.py**

## Step 6
The train.py file has been created such that we can give input values for various parameters and hyperparameters through command prompt while running to directly get the results. Train.py uses the NeuralNetwork.py file for implementing the Neural Network.
- **requirements: NeuralNetwork class from NeuralNetwork.py**

# Functionality of every file

## Module1.py:
This file loads the Fashion-MNIST dataset, selects one sample image per class, logs these images and their labels to Weights & Biases (W&B) as a table, and saves the table as a W&B artifact. It visualizes a representative sample of each fashion item category in the dataset.

## NeuralNetwork.py
This contains the Neural Network class which is to be further used for wandb sweeps as well as for train.py. 
- **Initialization:** The __init__ method sets up the network's architecture, including input and output sizes, layer configurations, and optimization parameters.
- **Layer Addition and Parameter Initialization:** The add_layer method appends layer sizes and activation functions. initialize_parameters sets up weights and biases using He, Xavier, or random initialization, and initializes optimizer-related variables.
- **Activation Functions and Forward Propagation:** The code provides implementations of ReLU, sigmoid, tanh, and softmax activation functions. The forward method performs forward propagation, storing intermediate results in a cache for backpropagation. It also handles Nesterov momentum's lookahead.
- **Loss Computation and Backpropagation:** The compute_loss method calculates cross-entropy or mean squared error loss. The backward method computes gradients using backpropagation and updates weights and biases according to the selected optimizer. It implements SGD, momentum, RMSprop, Adam, Nadam, and Nesterov momentum updates as asked in the assignment.
- **Training and Prediction:** The train_test_split function splits the dataset into training and validation sets. The train method trains the network using mini-batch gradient descent. It supports one-hot encoding for categorical labels and logs training and validation metrics to Weights & Biases (W&B). The predict method makes predictions by performing forward propagation and returning the class with the highest probability.
