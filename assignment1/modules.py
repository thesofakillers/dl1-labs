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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features: int, out_features: int, input_layer: bool = False):
        """
        Initializes the parameters of the module.

        Parameters:
        ----------
        in_features : int
            size of each input sample
        out_features : int
            size of each output sample
        input_layer : bool
            True if this is the first layer after the input, else False.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_features = in_features
        self.out_features = out_features
        self.params = {"bias": np.zeros(out_features)}
        if input_layer:
            # input_layer hasn't had ReLU applied yet, so kaiming init is different
            self.params["weight"] = np.random.normal(
                0, np.sqrt(1 / in_features), (out_features, in_features)
            )
        else:
            self.params["weight"] = np.random.normal(
                0, np.sqrt(2 / in_features), (out_features, in_features)
            )
        self.grads = {
            "weight": np.zeros_like(self.params["weight"]),
            "bias": np.zeros_like(self.params["bias"]),
        }
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x: np.ndarray):
        """
        Forward pass.

        Parameters
        ----------
        x: np.ndarray
            (in_features, ) array of input

        Returns
        -------
        out : np.ndarray
            (out_features, ) array of output
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x  # caching so we can use it in backward
        out = x @ self.params["weight"].T + self.params["bias"]
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Parameters
        ----------
        dout : np.ndarray
            (-1, out_features) array
            containing gradients of the previous module

        Returns
        -------
        dx: np.ndarray
            gradients with respect to the input of the module
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads["weight"] = dout.T @ self.x
        self.grads["bias"] = np.ones((1, dout.shape[0])) @ dout
        dx = dout @ self.params["weight"]
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: np.ndarray
            input to the module

        Returns
        -------
        out : np.ndarray
            result of applying ReLU to x.
            Same shape as x, since ReLU is element-wise.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        out = x * (x > 0).astype(float)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Parameters
        ----------
        dout : np.ndarray
            gradients of the previous module

        Returns:
        -------
        dx : np.ndarray
            gradients with respect to the input of the module
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * (1 * (self.x > 0).astype(float))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : np.ndarray
            input to the module

        Returns
        -------
        out: np.ndarray
            output of the module
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # need to use Max Trick to stabilize computation
        shift = np.max(x, axis=1)[:, np.newaxis]
        exps = np.exp(x - shift)
        out = exps / (exps.sum(axis=1)[:, np.newaxis])
        self.out = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Parameters
        ----------
        dout : np.ndarray
            gradients of the previous module

        Returns
        -------
        dx : np.ndarray
            gradients with respect to the input of the module
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        N = dout.shape[1]
        dx = self.out * (dout - (dout * self.out) @ np.ones((N, N)))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.out = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Parameters
        ----------
        x: np.ndarray
            input to the module
        y: np.ndarray
            labels of the input

        Returns
        -------
        out : float
            average cross entropy loss
            for this batch
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        one_hot = np.eye(y.max() + 1)[y]
        out = -(1 / x.shape[0]) * np.sum(one_hot * np.log(x))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Parameters
        ----------
        x : np.ndarray
            input to the module
        y : np.ndarray
            labels of the input.
            not one-hot encoded

        Returns
        -------
        dx : np.ndarray
            gradient of the loss with the respect to the input x.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        one_hot = np.eye(y.max() + 1)[y]
        dx = -1 / x.shape[0] * np.divide(one_hot, x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
