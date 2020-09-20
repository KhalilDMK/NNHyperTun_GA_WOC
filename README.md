# NNHyperTun_GA_WOC
Hyperparameter Tuning of Neural Networks: a Genetic Algorithms (GA) with Wisdom of Artificial Crowds (WOC) approach with GUI.

## Environment settings
We use Keras 2.2.4.

## Description
This repository includes code to optimize the hyperparameters of a Neural Network (NN) using a Genetic Algorithms approach with Wisdom of Crowds. We implement the approach on a simple regression problem on the <a href="https://keras.io/api/datasets/boston_housing/">"Boston Housing price regression dataset"</a>.

The model tuned is a Multilayer Perceptron (MLP).

## Hyperparameters
We tune the following hyperparameters:
* <b>Number of nodes: </b> constant for every hidden layer with values in {8, 16, 32, 64}.
* <b>Number of hidden layers: </b> with values in {2, 4, 5, 6, 8}.
* <b>Batch size: </b> with values in {8, 16, 32, 64, 128}.
* <b>Optimizer: </b> we try RMSProp, Adam, SGD, ADAGRAD, ADADELTA, Adamax and Nadam.
* <b>Activation function: </b> same for every hidden neuron. We try Rectified Linear Unit (relu), Exponential Linear Unit (elu), Hyperbolic Tangent (tanh) and Sigmoid.

## Crossover methods
We try two crossover methods:
* <b>Random cut-point crossover:</b>
Given the two parents, we select a random number between 1 and the number of genes that we will use as a cut-point. We select the first sub-sequence of genes, until the cut-point in the first parent, to constitute the first section of the crossover solution. Then, we add the second part of the second parent to the solution. This crossover technique is illustrated in the following figure:

<p align="center">
  <img src="https://github.com/KhalilDMK/NNHyperTun_GA_WOC/blob/master/Images/Random%20cutpoint%201.png" width="500">
</p>

* <b>Random section in the middle crossover:</b>
In this crossover technique, a sub-sequence of random length in the middle of the first parent is selected to go to the crossover child. Then, the first and third parts of the second parent are added to the solution. This technique is illustrated below:

<p align="center">
  <img src="https://github.com/KhalilDMK/NNHyperTun_GA_WOC/blob/master/Images/Random%20section%20in%20middle%201.png" width="500">
</p>
