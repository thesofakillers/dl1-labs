import typing as tg
import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from padding_numpy import zero_padding


def convolution2D(X, W, stride, padding):
    """
    A implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height IH and
    width IW. We convolve each input with F different filters, where each filter
    spans all C channels and has height FH and width FW.


    Parameters
    ----------
    X : npt.NDArray
        input data of shape (N, C, IH, IW)
    W : npt.NDArray
        Filter weight of shape (F, C, FH, FW)
    stride : tuple of int
        a tuple of 2 integer (sh, sw)
    padding : tuple int
        (ph, pw), amount of padding around each image on vert and hor dims

    Returns
    -------
    out : npt.NDArray
        Output data, of shape (N, F, OH, OW) where OH and OW given by
            OH= 1 + int ( (IH + 2*ph - FH)/ sh )
            OW= 1 + int ( (IW + 2*pw - FW)/ sw )
    """
    out: npt.NDArray[np.float64] = np.zeros(
        (
            X.shape[0],
            W.shape[0],
            int((X.shape[2] - W.shape[2] + 2 * padding[0]) / stride[0] + 1),
            int((X.shape[3] - W.shape[3] + 2 * padding[1]) / stride[1] + 1),
        ),
        dtype=np.float64,
    )

    X = zero_padding(X, padding)
    for b in range(out.shape[0]):
        for f in range(out.shape[1]):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    out[b, f, i, j] = np.sum(
                        X[
                            b,
                            :,
                            i * stride[0] : i * stride[0] + W.shape[2],
                            j * stride[1] : j * stride[1] + W.shape[3],
                        ]
                        * W[f, :, :, :],
                        dtype=np.float64,
                    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


if __name__ == "__main__":
    np.random.seed(1973)
    param1 = {
        "X": np.random.rand(2, 3, 23, 20),
        "W": np.random.rand(7, 3, 6, 6),
        "stride": (3, 6),
        "padding": (2, 3),
    }

    w_t = torch.from_numpy(param1["W"]).float()
    x_t = torch.from_numpy(
        np.pad(
            param1["X"], ((0, 0), (0, 0), (2, 2), (3, 3)), "constant", constant_values=0
        )
    ).float()
    conv = torch.nn.functional.conv2d(
        x_t, w_t, stride=param1["stride"], padding="valid"
    )
    conv = conv.cpu().detach().numpy()

    conv_numpy = convolution2D(**param1)

    assert conv.shape == conv_numpy.shape, "shape mismatch"
    print("Error :", (np.sum(conv - conv_numpy) ** 2))
    print("output shape :", conv_numpy.shape)
