################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import typing as tg
import numpy as np

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs: int, n_hidden: tg.List[int], n_classes: int):
        """
        Initializes MLP object.

        Parameters
        -----------
        n_inputs : int
            number of inputs.
        n_hidden : list of int
            specifies the number of units in each linear layer.
            If the list is empty, the MLP will not have any
            linear layers, and the model will simply perform
            a multinomial logistic regression.
        n_classes: int
            number of classes of the classification problem.
            This number is required in order to specify the
            output dimensions of the MLP
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.modules = []
        # getting all dims
        dims = [n_inputs] + n_hidden + [n_classes]
        # handle first layer specifically
        self.modules.append(LinearModule(n_inputs, dims[1], True))
        # handle hidden layers + output
        for i in range(1, len(dims) - 1):
            self.modules.append(ReLUModule())
            self.modules.append(LinearModule(dims[i], dims[i + 1]))
        # handle last layer (diff activation)
        self.modules.append(SoftMaxModule())
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x: np.ndarray):
        """
        Performs forward pass of the input.
        Here an input tensor x is transformed through
        several layer transformations.

        Parameters
        ----------
        x : np.ndarray
            (n_samples, n_input) input to the network

        Returns
        -------
        out : np.ndarray
            (n_samples, n_output) outputs of the network
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.reshape(x.shape[0], -1)
        for module in self.modules:
            x = module.forward(x)
        out = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout: np.ndarray):
        """
        Performs backward pass given the gradients of the loss.

        Parameters
        ----------
        dout : np.ndarray
            gradients of the loss
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for module in reversed(self.modules):
            dout = module.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for module in self.modules:
            module.clear_cache()
        #######################
        # END OF YOUR CODE    #
        #######################


# if __name__ == "__main__":
#     mlp = MLP(784, [], 10)
#     x = np.random.randn(7, 784)
#     y = mlp.forward(x)
#     # simulate backprop
#     dout = np.random.randn(7, 10)
#     mlp.backward(dout)
#     print(y.shape)
#     print(y)
#     print(y.sum(axis=1))
