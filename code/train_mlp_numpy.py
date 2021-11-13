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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Parameters
    ----------
    predictions : np.ndarray
        2D float array of size [batch_size, n_classes]
    labels : np.ndarray
        1D int array of size [batch_size]. Ground truth labels for
        each sample in the batch

    Returns
    -------
    accuracy : float
        the accuracy of predictions between 0 and 1,
        i.e. the average correct predictions over the whole batch
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    y_pred = predictions.argmax(axis=1)
    accuracy = np.mean(y_pred == targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_batches = len(data_loader)
    accuracies = np.zeros(n_batches)
    for i, (data, target) in enumerate(data_loader):
        y_pred = model.forward(data)
        accuracies[i] = accuracy(y_pred, target)
    avg_accuracy = accuracies.mean()
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Parameters
    ----------
    hidden_dims : list of ints
        specifying the hidden dimensionalities to use in the MLP.
    lr : float
        Learning rate of the SGD to apply.
    batch_size : int
        Minibatch size for the data loaders.
    epochs : int
        Number of training epochs to perform.
    seed : int
        Seed to use for reproducible results.
    data_dir : string
        Directory where to store/find the CIFAR10 dataset.

    Returns
    -------
    model : MLP
        An instance of 'MLP', the trained model that performed best on the validation set.
    val_accuracies : list of floats
        A list of scalar floats, containing the accuracies of the model on the
        validation set per epoch (element 0 - performance after epoch 1)
    test_accuracy: float
        average accuracy on the test dataset of the model that
        performed best on the validation. Between 0.0 and 1.0
    logging_info: dict
        An arbitrary object containing logging information. This is for you to
        decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # initializations
    candidate = MLP(np.array(cifar10["train"][0][0].shape).prod(), hidden_dims, 10)
    loss_module = CrossEntropyModule()
    logging_dict = {
        "loss": {"train": np.zeros(epochs), "validation": np.zeros(epochs)},
        "accuracy": {"train": np.zeros(epochs), "validation": np.zeros(epochs)},
    }
    best_accuracy = 0
    # training
    for epoch in range(epochs):
        for mode in ["train", "validation"]:
            n_batches = len(cifar10_loader[mode])  # for avg accuracy calculation
            with tqdm(cifar10_loader[mode], unit="batch") as curr_epoch:
                for features_X, true_y in curr_epoch:
                    curr_epoch.set_description(f"Epoch {epoch+1}: {mode}")
                    # forward pass and loss
                    y_pred = candidate.forward(features_X)
                    batch_loss = loss_module.forward(y_pred, true_y)
                    # backpropagation if in training mode
                    if mode == "train":
                        loss_grad = loss_module.backward(y_pred, true_y)
                        candidate.backward(loss_grad)
                        for module in candidate.modules[::2]:
                            module.params["weight"] -= lr * module.grads["weight"]
                            module.params["bias"] -= lr * module.grads["bias"]
                    # metrics
                    logging_dict["loss"][mode][epoch] += batch_loss / true_y.size
                    logging_dict["accuracy"][mode][epoch] += (
                        accuracy(y_pred, true_y) / n_batches
                    )
        # we use validation accuracy to pick the best model
        if mode == "validation":
            if logging_dict["accuracy"][mode][epoch] > best_accuracy:
                print(
                    f"New best accuracy: {logging_dict['accuracy'][mode][epoch]:0.3f}"
                )
                best_accuracy = logging_dict["accuracy"]["validation"][epoch]
                model = deepcopy(candidate)
    # additional return value requested
    val_accuracies = logging_dict["accuracy"]["validation"]
    # evaluated model on test set
    test_accuracy = evaluate_model(model, cifar10_loader["test"])
    #######################
    # END OF YOUR CODE    #
    #######################
    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help="Hidden dimensionalities to use inside the network. "
        'To specify multiple, use " " to separate them. Example: "256 128"',
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
