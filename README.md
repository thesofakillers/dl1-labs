# Deep Learning 1 Course - Practicals

This repository contains the code part of the three assignments of the Deep Learning 1 course, Fall 2021 edition.
I am omitting my University name for searchability reasons. My MSc university can be found on my LinkedIn or CV.

## Assignments

More details for each assignment can be found in the [assignment pdfs](./pdfs/).
For a brief overview, refer to the following:

1. Assignment 1: MLPs and Backpropagation. The following is implemented:
    - Differentiable Cross Entropy in NumPy
    - Differentiable Softmax in NumPy
    - Differentiable ReLU in NumPy
    - Differentiable Linear Layer in NumPy
    - A Multi-Layer Perceptron (MLP) in NumPy
    - An MLP in PyTorch
    - Training and Evaluation of both MLPs on CIFAR10
2. Assignment 2: CNNS, RNNs, and GNNs. The following is implemented:
    - Part 1: CNNs
        - Building blocks of a convolutional neural network in NumPy
            - Zero padding in NumPy
            - Differentiable convolution in NumPy 
            - Differentiable Max Pooling in NumPy
        - Training and evaluation of a number of torchvision models (ResNet-{18,34}, VGG-11, DenseNet-121)
    - Part 2: RNNs
        - LSTM in PyTorch, using only nn.Parameter and non-linear activation functions
        - Training and evaluation of generative LSTM Language Model on books.
    - Part 3: GNNs
        - Implementation of Graph Convolutional Neural Networks trained and evaluated on molecule data.
3. Assignment 3: Variational Autoencoders
    - Implementation of a Convolutional Variational Autoencoder in PyTorch
    - Training and Evaluation on FashionMNIST generation.
    
