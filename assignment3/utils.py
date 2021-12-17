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

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample
    from a distribution with the given mean and std

    Parameters
    ----------
    mean : torch.Tensor
        Tensor of arbitrary shape and range, denoting the mean of the distributions
    std : torch.Tensor
        Tensor of arbitrary shape with strictly positive values.
        Denotes the standard deviation of the distribution

    Returns
    -------
    z : torch.Tensor
        A sample of the distributions, with gradient support for both mean and std.
        The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), (
        "The reparameterization trick got a negative std as input. "
        + "Are you sure your input is std and not log_std?"
    )
    episolon = torch.randn_like(mean)
    z = mean + std * episolon
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given
    distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.

    Parameters
    ----------
    mean : torch.Tensor
        Tensor of arbitrary shape and range, denoting the mean of the distributions.
    log_std : torch.Tensor
        Tensor of arbitrary shape and range, denoting the
        log standard deviation of the distributions.

    Returns
    -------
    KLD : torch.Tensor
        Tensor with one less dimension than mean and log_std
        (summed over last dimension).
        The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    std = torch.exp(log_std)
    KLD = (std ** 2 + mean ** 2 - 1 - 2 * log_std).sum(dim=-1) / 2
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given
    by the ELBO into the bits per dimension score.

    Parameters
    ----------
    elbo : torch.Tensor
        Tensor of shape [batch_size]
    img_shape : array-like
        Shape of the input images, representing [batch, channels, height, width]

    Returns
    -------
    bpd : array-like
        The negative log likelihood in bits per dimension for the given image.
    """
    device = elbo.device
    bpd = (
        elbo
        * torch.log2(torch.full_like(elbo, np.e, device=device))
        * (1 / torch.prod(torch.tensor(img_shape[1:], device=device)))
    )
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space.
    The images in the manifold should represent the decoder's
    output means (not binarized samples of those).

    Parameters
    ----------
    decoder : torch.nn.Module
        Decoder model such as LinearDecoder or ConvolutionalDecoder.
    grid_size : int
        Number of steps/images to have per axis in the manifold.
        Overall you need to generate grid_size**2 images, and the distance
        between different latents in percentiles is 1/grid_size

    Returns
    -------
    img_grid : torch.Tensor
        Grid of images representing the manifold.
        of shape [grid_size, grid_size, channels, height, width]
    """
    z_values = torch.distributions.Normal(0, 1).icdf(
        torch.linspace(0.5 / grid_size, (grid_size - 0.5) / grid_size, grid_size)
    )
    grid_x, grid_y = torch.meshgrid(z_values, z_values, indexing="ij")

    z = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    z = z.to(decoder.device)

    # get decoder output, with shape (400, 16, 28, 28) and apply softmax
    recon_imgs = decoder(z)
    B, C, H, W = recon_imgs.size()
    probabilities = torch.softmax(recon_imgs, dim=1)
    # reshape so that these are accepted by multinomial
    probabilities = probabilities.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
    # sample from decoder output to get (1, 28, 28)
    samples = torch.multinomial(probabilities, 1)
    # reshape back into image
    samples = samples.reshape(B, 1, H, W)
    # and finally save our image to our grid
    img_grid = make_grid(
        samples.to(torch.float), nrow=grid_size, normalize=True, value_range=(0, 15)
    )

    return img_grid
