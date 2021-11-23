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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Parameters
    ----------
    predictions : torch.Tensor
        2D float array of size [batch_size, n_classes]
    targets : np.ndarray
        1D int array of size [batch_size]. Ground truth labels for
        each sample in the batch

    Returns
    -------
    accuracy : float
        the accuracy of predictions,
        i.e. the average correct predictions over the whole batch
    """
    # print(predictions.shape)
    # print(targets.shape)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    y_pred = predictions.argmax(dim=1)
    # y_true = targets.argmax(dim=1)
    y_true = targets
    accuracy = (y_pred == y_true).float().mean()
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
        data, target = data.to(model.device), target.to(model.device)
        predictions = model.forward(data)
        accuracies[i] = accuracy(predictions, target)
    avg_accuracy = accuracies.mean()
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(
    hidden_dims,
    lr,
    use_batch_norm,
    batch_size,
    epochs,
    seed,
    data_dir,
    save_path=None,
):
    """
    Performs a full training cycle of MLP model.

    Parameters
    ----------
    hidden_dims : list of int
        specificying the hidden dimensionalities to use in the MLP.
    lr : float
        Learning rate of the SGD to apply.
    use_batch_norm: bool
        If True, adds batch normalization layer into the network.
    batch_size : int
        Minibatch size for the data loaders.
    epochs : int
        Number of training epochs to perform.
    seed : int
        Seed to use for reproducible results.
    data_dir : string
        Directory where to store/find the CIFAR10 dataset.
    save_path : string, optional
        Path to serialize the loss and accuracy dictionary.

    Returns
    -------
    model : torch.nn.Module
        An instance of 'MLP', the trained model that performed
        best on the validation set.
    val_accuracies : list of float
        containing the accuracies of the model on the validation
        set per epoch (element 0 - performance after epoch 1)
    test_accuracy : float
        average accuracy on the test dataset of the model that
        performed best on the validation.
    logging_info : dict
        An arbitrary object containing logging information. This is for you to
        decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # initializations
    candidate = MLP(
        np.array(cifar10["train"][0][0].shape).prod(), hidden_dims, 10, use_batch_norm
    )
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(candidate.parameters(), lr)
    logging_info = {
        "loss": {"train": np.zeros(epochs), "validation": np.zeros(epochs)},
        "accuracy": {"train": np.zeros(epochs), "validation": np.zeros(epochs)},
    }
    best_accuracy = 0
    for epoch in range(epochs):
        for phase in ["train", "validation"]:
            n_batches = len(cifar10_loader[phase])
            if phase == "train":
                candidate.train()
            else:
                candidate.eval()
            with tqdm(cifar10_loader[phase], unit="batch") as curr_epoch:
                for features_X, true_y in curr_epoch:
                    curr_epoch.set_description(f"Epoch {epoch + 1}/{epochs}: {phase}")
                    # Move to GPU if possible
                    features_X = features_X.to(device)
                    true_y = true_y.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward pass and loss
                    y_pred = candidate.forward(features_X)
                    loss = loss_module(y_pred, true_y)
                    # backpropagation if in training mode
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    # metrics
                    logging_info["loss"][phase][epoch] += loss.item() / n_batches
                    logging_info["accuracy"][phase][epoch] += (
                        accuracy(y_pred, true_y) / n_batches
                    )
            # we use validation accuracy to pick the best model
            if phase == "validation":
                if logging_info["accuracy"][phase][epoch] > best_accuracy:
                    print(
                        f"New best accuracy: {logging_info['accuracy'][phase][epoch]:0.3f}"
                    )
                    best_accuracy = logging_info["accuracy"]["validation"][epoch]
                    model = deepcopy(candidate)
    # additional return value requested
    val_accuracies = logging_info["accuracy"]["validation"]
    # evaluated model on test set
    logging_info["accuracy"]["test"] = evaluate_model(model, cifar10_loader["test"])
    test_accuracy = logging_info["accuracy"]["test"]
    # serialize logging info if specified
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(logging_info, f)
    #######################
    # END OF YOUR CODE    #
    #######################
    return model, val_accuracies, test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use this option to add Batch Normalization layers to the MLP.",
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
    parser.add_argument(
        "--save-path",
        "-sp",
        default=None,
        type=str,
        help="Target path for serializing loss/accuracy dict",
    )
    args = parser.parse_args()
    kwargs = vars(args)


    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
