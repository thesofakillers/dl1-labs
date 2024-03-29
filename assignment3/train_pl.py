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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from fmnist import fmnist
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *


class VAE(pl.LightningModule):
    def __init__(self, num_filters, z_dim, lr):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Parameters
        ----------
        num_filters : int
            Number of channels to use in a CNN encoder/decoder
        z_dim : int
            Dimensionality of latent space
        lr : float
            Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
        self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.

        Parameters
        ----------
        imgs : array-like
            Batch of images of shape [B,C,H,W].
            The input images are converted to 4-bit, i.e. integers between 0 and 15.

        Returns
        -------
        L_rec : float
            The average reconstruction loss of the batch. Shape: single scalar
        L_reg : float
            The average regularization loss (KLD) of the batch. Shape: single scalar
        bpd : float
            The average bits per dimension metric of the batch.
            This is also the loss we train on. Shape: single scalar
        """

        mean, log_std = self.encoder(imgs)
        z = sample_reparameterize(mean, torch.exp(log_std))
        recon_imgs = self.decoder(z)

        L_reg = KLD(mean, log_std).mean()
        L_rec = (
            F.cross_entropy(recon_imgs, imgs.squeeze(), reduction="none")
            .sum(dim=(1, 2))
            .mean()
        )
        elbo = L_rec + L_reg
        bpd = elbo_to_bpd(elbo, imgs.size())

        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.

        Parameters
        ----------
        batch_size : int
            Number of images to generate

        Returns
        -------
        x_samples : array-like
            Sampled, 4-bit images. Shape: [B,C,H,W]
        """
        sampled_z = torch.randn(batch_size, self.hparams.z_dim)
        sampled_z = sampled_z.to(self.decoder.device)
        recon_imgs = self.decoder(sampled_z)
        B, C, H, W = recon_imgs.size()
        # shape B, C, H, W
        probabilities = F.softmax(recon_imgs, 1)
        # reshape to B*H*W, C
        probabilities = probabilities.permute((0, 2, 3, 1)).flatten(
            start_dim=0, end_dim=2
        )
        x_samples = torch.multinomial(probabilities, 1)
        # reshape to B, C, H, W
        x_samples = x_samples.reshape(B, 1, H, W)
        return x_samples

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("train_reconstruction_loss", L_rec, on_step=False, on_epoch=True)
        self.log("train_regularization_loss", L_reg, on_step=False, on_epoch=True)
        self.log("train_ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("train_bpd", bpd, on_step=False, on_epoch=True)

        return bpd

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("val_reconstruction_loss", L_rec)
        self.log("val_regularization_loss", L_reg)
        self.log("val_ELBO", L_rec + L_reg)
        self.log("val_bpd", bpd)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("test_bpd", bpd)


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Parameters
        ----------
        batch_size : int, default 64
            Number of images to generate
        every_n_epochs : int, default 5
            Only save those images every N epochs
            (otherwise tensorboard gets quite large)
        save_to_disk : bool, defualt False
            If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the sample_and_save function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch + 1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning "Trainer" object.
        pl_module : pl.LightningModule
            The VAE model that is currently being trained.
        epoch : int
            The epoch number to use for TensorBoard logging and saving of the files.
        """
        # sample images
        imgs = pl_module.sample(self.batch_size)
        # rearrange them into a grid
        img_grid = make_grid(
            imgs.to(torch.float),
            nrow=np.sqrt(self.batch_size).astype(int),
            normalize=True,
            value_range=(0, 15),
        )
        # save the grid to tensorboard
        trainer.logger.experiment.add_image("samples", img_grid, epoch)
        # save the grid to disk if desired
        if self.save_to_disk:
            save_image(img_grid, trainer.logger.log_dir + f"/samples_epoch_{epoch}.png")


def train_vae(args):
    """
    Function for training and testing a VAE model.

    Parameters
    ----------
    args
        Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = fmnist(
        batch_size=args.batch_size, num_workers=args.num_workers, root=args.data_dir
    )

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(
        save_weights_only=True, mode="min", monitor="val_bpd"
    )
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.epochs,
        callbacks=[save_callback, gen_callback],
        enable_progress_bar=args.progress_bar,
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )
    if not args.progress_bar:
        print(
            "[INFO] The progress bar has been suppressed. For updates on the training "
            f"progress, check the TensorBoard file at {trainer.logger.log_dir}. If you "
            'want to see the progress bar, use the argparse option "progress_bar".\n'
        )

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = VAE(num_filters=args.num_filters, z_dim=args.z_dim, lr=args.lr)

    # Training
    gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    # Testing
    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(
            img_grid,
            os.path.join(trainer.logger.log_dir, "vae_manifold.png"),
            normalize=False,
        )

    return test_result


if __name__ == "__main__":
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument(
        "--z_dim", default=20, type=int, help="Dimensionality of latent space"
    )
    parser.add_argument(
        "--num_filters",
        default=32,
        type=int,
        help="Number of channels/filters to use in the CNN encoder/decoder.",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument(
        "--data_dir",
        default="../data/",
        type=str,
        help="Directory where to look for the data."
        " For jobs on Lisa, this should be $TMPDIR.",
    )
    parser.add_argument("--epochs", default=80, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use in the data loaders."
        " To have a truly deterministic run, this has to be 0."
        " For your assignment report, you can use multiple workers"
        " (e.g. 4) and do not have to set it to 0.",
    )
    parser.add_argument(
        "--log_dir",
        default="VAE_logs",
        type=str,
        help="Directory where the PyTorch Lightning logs should be created.",
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help=(
            "Use a progress bar indicator for interactive experimentation. "
            "Not to be used in conjuction with SLURM jobs"
        ),
    )

    args = parser.parse_args()

    train_vae(args)
