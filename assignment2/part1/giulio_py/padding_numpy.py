import torch
import numpy as np
import numpy.typing as npt
import typing as tg
import matplotlib.pyplot as plt


def zero_padding(X: npt.NDArray, padding: tg.Tuple[int, int]):
    """
    Pad with zeros all images of the dataset X.
    The padding is applied to the height and width of an image.

    Parameters
    ----------
    X : npt.NDArray
        numpy array of shape (N, C, IH, IW) representing a batch of N images
    padding : tuple int
        (ph, pw), amount of padding around each image on vert and hor dims

    Returns
    -------
    zero_pad : npt.NDArray
        zero pad array of shape (N, C, IH + 2*ph, IW + 2*pw)
    """

    zero_pad: npt.NDArray = np.zeros(
        (
            X.shape[0],
            X.shape[1],
            X.shape[2] + 2 * padding[0],
            X.shape[3] + 2 * padding[1],
        ),
        dtype=X.dtype,
    )
    zero_pad[:, :, padding[0] : -padding[0], padding[1] : -padding[1]] = X
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return zero_pad


if __name__ == "__main__":
    # test zero_padding function
    np.random.seed(1968)

    x = np.random.rand(2, 3, 4, 4)
    padding = (3, 2)
    x_pad = zero_padding(x, padding)

    assert x_pad.shape == (
        x.shape[0],
        x.shape[1],
        x.shape[2] + 2 * padding[0],
        x.shape[3] + 2 * padding[1],
    )
    assert np.all(
        x_pad[
            :,
            :,
            padding[0] : padding[0] + x.shape[2],
            padding[1] : padding[1] + x.shape[3],
        ]
        == x
    )

    print("your implementation is correct")
    print("shape of x is :", x.shape)
    print("shape of x_pad is :", x_pad.shape)

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title("x")
    axarr[0].imshow(x[0, 0, :, :])
    axarr[1].set_title("x_pad")
    axarr[1].imshow(x_pad[0, 0, :, :])
    plt.show()
