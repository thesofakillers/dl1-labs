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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to
    try out different plotting configurations without re-running your models every time.

    Parameters
    ----------
    results_filename : string
        specifies the name of the file to which the results
        should be saved.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # define our hyperparameter search space
    dim_set = [[128], [256, 128], [512, 256, 128]]
    bn_set = [False, True]
    # initialize results dictionary
    results = {bn_bool: {tuple(dims): {} for dims in dim_set} for bn_bool in bn_set}
    # # loop over all hyperparameter configurations
    for dims in dim_set:
        for bn_bool in bn_set:
            print(f"Training model with dims: {dims}, batch norm: {bn_bool}")
            # only intereseted in logging_dict
            _, _, _, logging_dict = train_mlp_pytorch.train(
                dims, 0.1, bn_bool, 128, 20, 42, "data/"
            )
            results[bn_bool][tuple(dims)] = logging_dict
    # # serialize results
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Parameters
    ----------
    results_filename : string
        which specifies the name of the file from which the results
        are loaded.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, "rb") as f:
        results = pickle.load(f)
    # define our hyperparameter search space (we could derive this from the dict)
    dim_set = [[128], [256, 128], [512, 256, 128]]
    # bn_set = [False, True]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey="row")
    epochs = np.arange(1, 21)
    for i, (phase, row) in enumerate(zip(("train", "validation"), axes)):
        for dims, ax in zip(dim_set, row):
            ax.plot(
                epochs,
                results[False][tuple(dims)]["accuracy"][phase],
                label="No Batch Norm",
                color="black" if phase == "train" else "darkblue",
                linestyle="solid",
            )
            ax.plot(
                epochs,
                results[True][tuple(dims)]["accuracy"][phase],
                label="With Batch Norm",
                color="black" if phase == "train" else "darkblue",
                linestyle="dashed",
            )
            ax.set_xlabel("Epoch")
            ax.set_title(
                f"{'Training' if phase =='train' else 'Validation'} Accuracy per Epoch"
                f"\nHidden Layer Dimensionality: {dims}"
            )
        ax.legend()
    fig.suptitle(
        "Validation and Training Accuracy per Epoch with and without Batch"
        " Normalization\nfor MLP's with different hidden layer dimensionalities"
    )
    fig.set_tight_layout(True)
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    # Feel free to change the code below as you need it.
    FILENAME = "output/compare_results.pkl"
    # if not os.path.isfile(FILENAME):
    #     train_models(FILENAME)
    plot_results(FILENAME)
