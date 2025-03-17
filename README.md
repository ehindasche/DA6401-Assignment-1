# DA6401-Assignment-1
This project involves the development and implementation of neural network architectures from fundamental principles using NumPy. The Fashion-MNIST dataset serves as the training and evaluation corpus. Comprehensive experiment tracking and results visualization are achieved through the integration of Weights & Biases (wandb.ai), facilitating detailed reporting and insightful analysis.

## Step 1
Module1.py contains the code for Question 1 of the assignment. 

## Step 2
NeuralNetwork.py contains code for Question 2-3. It contains code for the entire Neural Network class with necessary functions and optimizers included in the class.

## Step 3
wandb.py contains code for sweep functionality. The charts obtained in Question 4-6 and also the ones for "MSE" in question 8 have been obtained through this.

## Step 4
confusion_matrix.py generates the confusion matrix. It uses code from the Module_3.py file.
The file was run by training the NeuralNetwork on the best parameters obtained (mentioned in the report)

## Step 5
The mnist.py file uses the sweep functionality to run sweeps based on only 3 parameters. This is mentioned in the report along with the charts attached.

## Step 6
The train.py file has been created such that we can give input values for various parameters and hyperparameters through command prompt while running to directly get the results. Train.py uses the NeuralNetwork.py file for implementing the Neural Network.
