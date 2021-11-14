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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Parameters
        ----------
        n_inputs : int
            number of inputs.
        n_hidden : list of int
            specifies the number of units in each linear layer.
            If the list is empty, the MLP will not have any
            linear layers, and the model will simply
            perform a multinomial logistic regression.
        n_classes : int
            number of classes of the classification problem.
            This number is required in order to specify the
            output dimensions of the MLP
        use_batch_norm : bool
            If True, add a Batch-Normalization layer in between
            each Linear and ReLU layer.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        # getting all dims
        dims = [n_inputs] + n_hidden + [n_classes]
        # first layer outside loop so that loop ends with linear
        self.layers = nn.ModuleList([nn.Linear(dims[0], dims[1])])
        for i in range(1, len(dims) - 1):
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Parameters
        ----------
        x : torch.Tensor
            input to the network
        Returns
        -------
        out : torch.Tensor
            outputs of the network
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        out = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device


# if __name__ == "__main__":
#     import torch
#     import numpy as np

#     mlp = MLP(784, [3, 5, 6], 10, True)
#     x = np.random.randn(7, 784)
#     y = mlp.forward(torch.Tensor(x))

#     print(f"forward output shape: {y.shape}")
#     print(f"forward output: {y}")
#     print(f"forward output sum over axis 1: {y.sum(axis=1)}")
#     print(f"mlp layers: {mlp.layers}")
#     print(mlp.modules)
