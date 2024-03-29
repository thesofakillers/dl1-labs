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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initializes MLP object for (multivariate) regression

        Parameters
        ----------
        n_inputs : int
            number of inputs.
        n_hidden : list of int
            specifies the number of units in each linear layer.
            If the list is empty, the MLP will not have any
            linear layers, and the model will simply
            perform a multinomial logistic regression.
        n_outputs : int
            This number is required in order to specify the
            output dimensions of the MLP
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        # getting all dims
        dims = [n_inputs] + n_hidden + [n_outputs]
        # first layer outside loop so that loop ends with linear
        self.layers = nn.ModuleList([nn.Linear(dims[0], dims[1])])
        for i in range(1, len(dims) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
            x: input to the network
        Returns:
            out: outputs of the network
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


class GNN(nn.Module):
    """
    implements a graph neural network in pytorch.
    In particular, we will use pytorch geometric's
    nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        num_convolution_blocks: int,
    ) -> None:
        """
        Initializes a GNN with the following structure:
        node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear

        Parameters
        ----------
        n_node_features : int
            number of input features on each node
        n_edge_features : int
            number of input features on each edge
        n_hidden : int
            number of hidden features within the neural networks
            (embeddings, nodes after graph convolutions, etc.)
        n_output : int
            how many output features
        num_convolution_blocks : int
            how many blocks convolutions should be performed.
            A block may include multiple convolutions
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super(GNN, self).__init__()
        self.embedding = nn.Linear(n_node_features, n_hidden)
        self.GNN = nn.ModuleList([])
        for i in range(num_convolution_blocks):
            conv_block = nn.ModuleList(
                [
                    nn.ReLU(),
                    geom_nn.RGCNConv(n_hidden, n_hidden, n_edge_features),
                    nn.ReLU(),
                    geom_nn.MFConv(n_hidden, n_hidden),
                ]
            )
            self.GNN.append(conv_block)
        self.head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features per node
        edge_index : torch.Tensor
            List of vertex index pairs representing the edges in the graph
            (PyTorch geometric notation)
        edge_attr : torch.Tensor
            edge attributes (pytorch geometric notation)
        batch_idx : torch.Tensor
            Index of batch element for each node

        Returns
        -------
        prediction : torch.Tensor
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embedding(x)
        for block in self.GNN:
            for layer in block:
                if isinstance(layer, geom_nn.RGCNConv):
                    x = layer(x, edge_index, edge_attr)
                elif isinstance(layer, geom_nn.MFConv):
                    x = layer(x, edge_index)
                else:
                    x = layer(x)
        x = geom_nn.global_add_pool(x, batch_idx)
        out = self.head(x)
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
