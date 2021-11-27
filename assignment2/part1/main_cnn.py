###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
from copy import deepcopy
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from tqdm.auto import tqdm

from augmentations import (
    gaussian_noise_transform,
    gaussian_blur_transform,
    contrast_transform,
    jpeg_transform,
)
from cifar10_utils import get_train_validation_set, get_test_set
from utils import Suppress


def get_accuracy(predictions, targets):
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
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    y_pred = predictions.argmax(dim=1)
    accuracy = (y_pred == targets).float().mean()
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, num_classes=10):
    """
    Returns the model architecture for the provided model_name.

    Parameters
    ----------
    model_name : string
        Name of the model architecture to be returned.
        Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18',
                    'resnet34', 'densenet121']
        All models except debug are taking from the torchvision library.
    num_classes : int
        Number of classes for the final layer (for CIFAR10 by default 10)

    Returns
    -------
    cnn_model : nn.Module
        object representing the model architecture.
    """
    if model_name == "debug":  # Use this model for debugging
        cnn_model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, num_classes))
    elif model_name == "vgg11":
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == "vgg11_bn":
        cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == "resnet18":
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == "densenet121":
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture "{model_name}"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Parameters
    ----------
    model : nn.Module
        Model architecture to train.
    lr : float
        Learning rate to use in the optimizer.
    batch_size : int
        Batch size to train the model with.
    epochs : int
        Number of epochs to train the model for.
    data_dir : string
        Directory where the CIFAR10 dataset should be loaded from or downloaded to.
    checkpoint_name : string
        Filename to save the best model on validation to.
    device : torch.device
        Device to use for training.

    Returns
    -------
    model : nn.Module
        Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)
    data_loader = {
        "train": data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        ),
        "val": data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    }
    # Initialize the optimizers and learning rate scheduler.
    # We provide a recommend setup, which you are allowed to change if interested.
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[90, 135], gamma=0.1
    )

    # Training loop with validation after each epoch.
    # Save the best model, and remember to use the lr scheduler.
    best_state_dict = None
    best_accuracy = 0
    for epoch in range(epochs):
        epoch_val_acc = 0
        for phase in ["train", "val"]:
            n_batches = len(data_loader[phase])
            if phase == "train":
                model.train()
            else:
                model.eval()
            with tqdm(data_loader[phase], unit="batch") as curr_epoch:
                for features_X, true_y in curr_epoch:
                    curr_epoch.set_description(f"Epoch {epoch + 1}/{epochs}: {phase}")
                    # Move to GPU if possible
                    features_X = features_X.to(device)
                    true_y = true_y.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward pass and loss
                    y_pred = model.forward(features_X)
                    loss = loss_module(y_pred, true_y)
                    # backpropagation if in training mode
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    if phase == "val":
                        epoch_val_acc += get_accuracy(y_pred, true_y) / n_batches
            if phase == "train":
                scheduler.step()
            if phase == "val":
                if epoch_val_acc > best_accuracy:
                    best_accuracy = epoch_val_acc
                    best_state_dict = deepcopy(model.state_dict())
    # Save the best model to disk
    with open(checkpoint_name, "wb") as f:
        torch.save(best_state_dict, f)
    # Load best model and return it.
    model.load_state_dict(best_state_dict)

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Parameters
    model : nn.Module Model instance to evaluate.
    data_loader: data.DataLoader
        The data loader of the dataset to evaluate on.
    device : torch.device
        Device to use for training.

    Returns
    -------
    accuracy : float
        The average accuracy on the dataset.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_batches = len(data_loader)
    accuracies = np.zeros(n_batches)
    for i, (features_X, target) in enumerate(data_loader):
        features_X, target = features_X.to(device), target.to(device)
        predictions = model.forward(features_X)
        accuracies[i] = get_accuracy(predictions, target)
    accuracy = accuracies.mean()
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Parameters
    ----------
    model : nn.Module
        trained model instance to test.
    batch_size : int
        Batch size to use in the test.
    data_dir : string
        Directory where the CIFAR10 dataset should be loaded from or downloaded to.
    device : torch.device
        Device to use for training.
    seed : int
        The seed to set before testing to ensure a reproducible test.

    Returns
    -------
    test_results : dict
        Dictionary containing an overview of the
        accuracies achieved on the different
        corruption functions and the plain test set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    severities = range(1, 6)
    aug_funcs = [
        gaussian_noise_transform,
        gaussian_blur_transform,
        contrast_transform,
        jpeg_transform,
    ]
    aug_func_names = ["gaussian_noise", "gaussian_blur", "contrast", "jpeg"]
    test_results = {name: np.zeros(len(severities)) for name in aug_func_names}

    # test model on each of the corruption functions
    for aug_func, aug_func_name in zip(aug_funcs, aug_func_names):
        print(f"Testing {aug_func_name}")
        for s, severity in enumerate(tqdm(severities, unit="severity")):
            with Suppress(suppress_stdout=True):
                test_set = get_test_set(data_dir, aug_func(severity))
                test_loader = data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
            test_results[aug_func_name][s] = evaluate_model(model, test_loader, device)
    # repeat the test on the plain test set
    print("Testing on plain test set")
    test_set = get_test_set(data_dir)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )
    test_results["plain"] = evaluate_model(model, test_loader, device)
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Parameters
    ----------
    model_name : string
        Model architecture to test.
    batch_size : int
        Batch size to use in the test.
    data_dir : string
        Directory where the CIFAR10 dataset should be loaded from or downloaded to.
    device : torch.device
        Device to use for training.
    seed : int
        The seed to set before testing to ensure a reproducible test.

    Returns
    -------
    test_results : dict
        Dictionary containing an overview of the
        accuracies achieved on the different
        corruption functions and the plain test set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    # instantiate untrained model
    model = get_model(model_name, 10)
    model = model.to(device)
    save_name = f"{model_name}_lr{lr}_bs{batch_size}_e{epochs}_s{seed}"
    checkpoint_name = f"cpt_{save_name}.pth"
    # check if model already trained
    already_trained: bool = os.path.isfile(checkpoint_name)
    if already_trained:
        print(f"Loading {model_name} model from checkpoint")
        # load the pretrained model
        model.load_state_dict(torch.load(checkpoint_name))
    else:
        print(f"Training model {model_name}")
        # train the model and save state dict to disk
        model = train_model(
            model, lr, batch_size, epochs, data_dir, checkpoint_name, device
        )
    print(f"Testing model {model_name}")
    # test model using the test_model function
    test_results = test_model(model, batch_size, data_dir, device, seed)
    # save the results to disk
    print("Testing complete. Saving results to disk")
    with open(f"results_{save_name}.pkl", "wb") as f:
        pickle.dump(test_results, f)
    return test_results
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need,
    e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model_name", default="debug", type=str, help="Name of the model to train."
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=150, type=int, help="Max number of epochs")
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
    main(**kwargs)
