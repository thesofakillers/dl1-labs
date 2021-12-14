################################################################################
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
# Date Created: 2020-11-27
################################################################################

import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from fmnist import fmnist
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *


class VAE(nn.Module):

    def __init__(self, num_filters, z_dim, *args,
                 **kwargs):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
        self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

    def forward(self, imgs):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]. 
                   The input images are converted to 4-bit, i.e. integers between 0 and 15.
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # Hints: 
        # - Implement the empty functions in utils.py before continuing
        # - The forward run consists of encoding the images, sampling in
        #   latent space, and decoding.
        # - You might find loss functions defined in torch.nn.functional 
        #   helpful for the reconstruction loss

        L_rec = None
        L_reg = None
        bpd = None
        raise NotImplementedError
        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, 4-bit images. Shape: [B,C,H,W]
        """
        x_samples = None
        raise NotImplementedError
        return x_samples

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.decoder.device


def sample_and_save(model, epoch, summary_writer, batch_size=64):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    # Hints:
    # - You can access the logging directory path via summary_writer.log_dir
    # - Remember converting the 4-bit images to a format recognized by tensorboard, e.g. float values between 0 and 1.
    # - Use the torchvision function "make_grid" to create a grid of multiple images
    # - Use the torchvision function "save_image" to save an image grid to disk

    raise NotImplementedError

@torch.no_grad()
def test_vae(model, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    average_bpd = None
    average_rec_loss = None
    average_reg_loss = None
    raise NotImplementedError
    return average_bpd, average_rec_loss, average_reg_loss


def train_vae(model, train_loader, optimizer):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    average_bpd = None
    average_rec_loss = None
    average_reg_loss = None
    raise NotImplementedError
    return average_bpd, average_rec_loss, average_reg_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare logging
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_writer = SummaryWriter(experiment_dir)
    if not args.progress_bar:
        print("[INFO] The progress bar has been suppressed. For updates on the training " + \
              f"progress, check the TensorBoard file at {experiment_dir}. If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Load dataset
    train_loader, val_loader, test_loader = fmnist(batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   root=args.data_dir)

    # Create model
    model = VAE(num_filters=args.num_filters,
                z_dim=args.z_dim,
                lr=args.lr)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Sample image grid before training starts
    sample_and_save(model, 0, summary_writer, 64)

    # Tracking variables for finding best model
    best_val_bpd = float('inf')
    best_epoch_idx = 0
    print(f"Using device {device}")
    epoch_iterator = (trange(1, args.epochs + 1, desc=f"{args.model} VAE")
                      if args.progress_bar else range(1, args.epochs + 1))
    for epoch in epoch_iterator:
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        epoch_train_bpd, train_rec_loss, train_reg_loss = train_vae(
            model, train_iterator, optimizer)

        # Validation epoch
        val_iterator = (tqdm(val_loader, desc="Testing", leave=False)
                        if args.progress_bar else val_loader)
        epoch_val_bpd, val_rec_loss, val_reg_loss = test_vae(model, val_iterator)

        # Logging to TensorBoard
        summary_writer.add_scalars(
            "BPD", {"train": epoch_train_bpd, "val": epoch_val_bpd}, epoch)
        summary_writer.add_scalars(
            "Reconstruction Loss", {"train": train_rec_loss, "val": val_rec_loss}, epoch)
        summary_writer.add_scalars(
            "Regularization Loss", {"train": train_reg_loss, "val": val_reg_loss}, epoch)
        summary_writer.add_scalars(
            "ELBO", {"train": train_rec_loss + train_reg_loss, "val": val_rec_loss + val_reg_loss}, epoch)

        if epoch % 5 == 0:
            sample_and_save(model, epoch, summary_writer, 64)

        # Saving best model
        if epoch_val_bpd < best_val_bpd:
            best_val_bpd = epoch_val_bpd
            best_epoch_idx = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

    # Load best model for test
    print(f"Best epoch: {best_epoch_idx}. Load model for testing.")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))

    # Test epoch
    test_loader = (tqdm(test_loader, desc="Testing", leave=False)
                   if args.progress_bar else test_loader)
    test_bpd, _, _ = test_vae(model, test_loader)
    print(f"Test BPD: {test_bpd}")
    summary_writer.add_scalars("BPD", {"test": test_bpd}, best_epoch_idx)

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(img_grid, os.path.join(experiment_dir, 'vae_manifold.png'),
                   normalize=False)

    return test_bpd


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/', type=str, 
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    main(args)

