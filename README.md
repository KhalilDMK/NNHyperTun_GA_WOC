# NNHyperTun_GA_WOC
Hyperparameter Tuning of Neural Networks: a Genetic Algorithms (GA) with Wisdom of Artificial Crowds (WOC) approach with GUI.

## Environment settings
We use Keras 2.2.4.

## Description
This repository includes code to optimize the hyperparameters of a Neural Network (NN) using a Genetic Algorithms approach with Wisdom of Crowds. We implement the approach on a simple regression problem on the <a href="https://keras.io/api/datasets/boston_housing/">"Boston Housing price regression dataset"</a>.

The model tuned is a Multilayer Perceptron (MLP).

## Hyperparameters
We tune the following hyperparameters:
* <br>Number of nodes: </br> constant for every hidden layer with values in {8, 16, 32, 64}.
* <br>Number of hidden layers: </br> with values in {2, 4, 5, 6, 8}.
* <br>Batch size: </br> with values in {8, 16, 32, 64, 128}.
* <br>Optimizer: </br> we try RMSProp, Adam, SGD, ADAGRAD, ADADELTA, Adamax and Nadam.
* <br>Activation function: </br> same for every hidden neuron. We try Rectified Linear Unit (relu), Exponential Linear Unit (elu), Hyperbolic Tangent (tanh) and Sigmoid.

## Crossover methods
We try two crossover methods:
* <br>Random cut-point crossover:</br>
Given the two parents, we select a random number between 1 and the number of genes that we will use as a cut-point. We select the first sub-sequence of genes, until the cut-point in the first parent, to constitute the first section of the crossover solution. Then, we add the second part of the second parent to the solution. This crossover technique is illustrated in the following figure:

![alt text](https://drive.google.com/file/d/1lqr1OzeD-fleY0JVtG299e-Kvm5j5B3b/view?usp=sharing)
