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

import pickle
import os
import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *
from networks import *


def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Parameters
    ----------
    molecules : Batch
        batch of molecules from pytorch geometric

    Returns
    -------
    permuted : Batch
        the molecules with permuted atoms
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """
    Performs forward pass and computes loss,
    adjusting process based on model type

    Parameters
    ----------
    model : nn.Module
        trainable network
    molecules : Batch
        batch of molecules from pytorch geometric
    criterion : Callable
        callable which takes a prediction and the ground truth

    Returns
    -------
    loss : torch.Tensor
        scalar tensor of the loss
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    true_y = get_labels(molecules)
    true_y = true_y.to(model.device)
    if isinstance(model, MLP):
        # massage batch into corect format for MLP
        features_X = get_mlp_features(molecules)
        features_X = features_X.to(model.device)
        # forward
        pred_y = model(features_X).squeeze()
    elif isinstance(model, GNN):
        features_X = get_node_features(molecules)
        features_X = features_X.to(model.device)
        # forward
        pred_y = model(
            features_X,
            molecules.edge_index,
            molecules.edge_attr.argmax(dim=-1),
            molecules.batch,
        ).squeeze()
    # compute loss
    loss = criterion(pred_y, true_y)
    #######################
    # END OF YOUR CODE    #
    #######################
    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Parameters
    model : nn.Module
        trainable network
    data_loader: DataLoader
        The data loader of the dataset to evaluate
    criterion : Callable
        loss module, i.e. torch.nn.MSELoss()
    permute : bool
        whether to permute the atoms within a molecule

    Returns
    -------
    avg_loss : float
        the average loss of the model on the dataset.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    n_batches = len(data_loader)
    losses = np.zeros(n_batches)
    for i, molecule in enumerate(data_loader):
        if permute:
            molecule = permute_indices(molecule)
        losses[i] = compute_loss(model, molecule, criterion)
    avg_loss = losses.mean()
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """
    a full training cycle of an mlp / gnn on qm9.

    Parameters
    ----------
    model : nn.Module
        a differentiable pytorch module which estimates the U0 quantity
    lr : float
        learning rate of optimizer
    batch_size : int
        batch size of molecules
    epochs : int
        number of epochs to optimize over
    seed : int
        random seed
    data_dir : str
        where to place the qm9 data

    Returns
    -------
    model : nn.Module
        the trained model which performed best on the validation set
    test_loss : float
        the loss over the test set
    permuted_test_loss : float
        the loss over the test set where atomic indices have been permuted
    val_losses : array-like
        the losses over the validation set at every epoch
    logging_info : dict
        general object with information for making plots or whatever you'd like to do with it
    """
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Loading the dataset
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # collect dataloaders in dictionary for easy access
    data_loaders = {
        "train": train_dataloader,
        "val": valid_dataloader,
        "test": test_dataloader,
    }
    criterion = torch.nn.MSELoss().to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    best_loss = float("inf")
    best_model: nn.Module
    logging_info = {
        "loss": {
            "val": np.zeros(epochs),
            "test": {"regular": None, "permuted": None},
        }
    }
    for epoch in range(epochs):
        for phase in ["train", "val"]:
            # turn on training mode accordingly
            n_batches = len(data_loaders[phase])
            model.train(phase == "train")
            with tqdm(data_loaders[phase], unit="batch") as curr_epoch:
                for molecule_batch in curr_epoch:
                    curr_epoch.set_description(f"Epoch {epoch + 1}/{epochs}: {phase}")
                    # zero the gradients
                    optimizer.zero_grad()
                    # forward and compute loss
                    loss = compute_loss(model, molecule_batch, criterion)
                    # backpropagation if in training mode
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    if phase == "val":
                        logging_info["loss"]["val"][epoch] += loss.item() / n_batches
            # check if our model is the best so far
            if phase == "val":
                if logging_info["loss"]["val"][epoch] < best_loss:
                    print(f"New best loss: {logging_info['loss'][phase][epoch]:0.6f}")
                    best_loss = logging_info["loss"]["val"][epoch]
                    best_model = deepcopy(model)
    # store validation loss in its own variable
    val_losses = logging_info["loss"]["val"]
    # Test best model with and without permutation
    logging_info["loss"]["test"]["regular"] = evaluate_model(
        best_model, data_loaders["test"], criterion, permute=False
    )
    logging_info["loss"]["test"]["permuted"] = evaluate_model(
        best_model, data_loaders["test"], criterion, permute=True
    )
    # store these evaluation losses in their own variables
    test_loss = logging_info["loss"]["test"]["regular"]
    permuted_test_loss = logging_info["loss"]["test"]["permuted"]
    # they want the best model to be in the variable 'model'
    model = deepcopy(best_model)
    #######################
    # END OF YOUR CODE    #
    #######################
    return model, test_loss, permuted_test_loss, val_losses, logging_info


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    # check if we've already trained
    model.to(device)
    model, test_loss, permuted_test_loss, val_losses, logging_info = train(
        model, **kwargs
    )
    # serialize logging info
    with open(f"{which_model}_results.pkl", "wb") as f:
        pickle.dump(logging_info, f)
    # report metrics
    print(f"Test Loss: {test_loss}")
    print(f"Permuted Test Loss: {permuted_test_loss}")
    print(f"Validation Losses: {val_losses}")
    # plot the loss curve, etc. below.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(val_losses) + 1)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", color="black")
    plt.ylabel("Average Epoch Loss")
    plt.xlabel("Epoch Number")
    plt.legend()
    plt.title(f"Validation Loss of {'MLP' if which_model =='mlp' else 'GNN'} model")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="mlp",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
