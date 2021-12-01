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
# Date Adapted: 2021-11-11
###############################################################################

import pprint
from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


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


def sample(args):
    """
    Samples sentences from a trained TextGenerationModel

    to save the print statements to a file, run
    `python train.py --sample --txt_file=<path_to_text_file> > outputfile`
    """
    set_seed(args.seed)
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.vocabulary_size = dataset._vocabulary_size
    book_name = args.txt_file.split("/")[-1].split(".")[0]
    print("Hyperparameters:")
    print("##################")
    pprint.pprint(vars(args))
    print("##################")
    for epoch in [1, 5, 20]:
        # initialize model at each epoch to ensure h and c are reset
        model = TextGenerationModel(args).to(args.device)
        print(f"{book_name}: Epoch {epoch}")
        # load relevant model checkpoint
        checkpoint_path = f"{args.checkpoint_dir}{book_name}-lstm-e{epoch}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        # sample a batch of 5
        text_samples = model.sample(5, args.input_seq_length, args.temperature)
        for i, text_sample in enumerate(text_samples.T):
            # need to convert each batch to its string representation
            string_rep = dataset.convert_to_string(text_sample.tolist())
            print(f"\nSample {i+1}/5:")
            print("----------")
            print(f"{string_rep}")
            print("----------")
        print("==================")


def train(args):
    """
    Trains an LSTM model on a text dataset

    Parameters
    args : Namespace
        object of the command line arguments as
        specified in the main function.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    # needed for our TextGenerationModel
    args.vocabulary_size = dataset._vocabulary_size
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=text_collate_fn,
    )
    # initialization
    model = TextGenerationModel(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_module = nn.CrossEntropyLoss().to(args.device)
    writer = SummaryWriter()
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        n_batches = len(data_loader)
        with tqdm(data_loader, unit="batch") as curr_epoch:
            for features_X, true_y in curr_epoch:
                curr_epoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
                # move to GPU if available
                features_X = features_X.to(args.device)
                true_y = true_y.to(args.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                y_pred = model.forward(features_X)
                # reshape sequence tensors for loss
                true_y = true_y.view(-1)
                y_pred = y_pred.view(-1, y_pred.shape[-1])
                # calculate and record loss and accuracy
                loss = loss_module(y_pred, true_y)
                epoch_loss += loss.item() / n_batches
                epoch_accuracy += get_accuracy(y_pred, true_y) / n_batches
                # # backward pass
                loss.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                # update parameters
                optimizer.step()
                # reset hidden/cell states
                model.lstm.h = None
                model.lstm.c = None
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
            # generate sentences at different stages of training
            if epoch in {0, 4, args.num_epochs - 1}:
                # save a checkpoint of the model, for sampling from later
                book_name = args.txt_file.split("/")[-1].split(".")[0]
                checkpoint_path = (
                    f"{args.checkpoint_dir}{book_name}-lstm-e{epoch+1}.pth"
                )
                with open(checkpoint_path, "wb") as f:
                    torch.save(model.state_dict(), f)
    writer.flush()
    writer.close()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument(
        "--txt_file", type=str, required=True, help="Path to a .txt file to train on"
    )
    parser.add_argument(
        "--input_seq_length", type=int, default=30, help="Length of an input sequence"
    )
    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=1024,
        help="Number of hidden units in the LSTM",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=256,
        help="Dimensionality of the embeddings.",
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to train with."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=5.0, help="Gradient clipping norm"
    )

    # Additional arguments. Feel free to add more arguments
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for pseudo-random number generator"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Sample from the model instead of training."
        " Requires checkpoints at epoch 1, 5, and 20",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for softmax sampling from the model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-cd",
        type=str,
        default="./",
        help="path to directory containing checkpoints, including final forward slash",
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available, else use CPU
    if args.sample:
        sample(args)
    else:
        train(args)
