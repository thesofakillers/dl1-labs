import typing as tg
import numpy as np
import numpy.typing as npt
import torch


def convolution2D_backward_input(
    out_grad: npt.NDArray, X: npt.NDArray, W: npt.NDArray, stride: tg.Tuple[int, int]
) -> npt.NDArray:
    """
    A implementation of the backward pass for a convolutional layer.

    Parameters
    ----------
    out_grad  : npt.NDArray
        gradient of the Loss with respect to the output of
        the conv layer with shape (N, F, OW, OH)
    X : npt.NDArray
        input data of shape (N, C, IH, IW)
    W : npt.NDArray
        Filter weight of shape (F, C, FH, FW)
    stride : tuple of int
        [sh, sw]

    Returns
    -------
    dX : npt.NDArray
        Gradient with respect to X
    """

    dX = None
    ###########################################################################
    # Implement the convolutional backward pass.                              #
    ###########################################################################
    dX = np.zeros_like(X, dtype=np.float64)
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            for k in range(X.shape[2]):
                for g in range(X.shape[3]):
                    for f in range(out_grad.shape[1]):
                        for i in range(out_grad.shape[2]):
                            for j in range(out_grad.shape[3]):
                                k_index = k - stride[0] * i
                                g_index = g - stride[1] * j
                                if (
                                    0 <= k_index < W.shape[2]
                                    and 0 <= g_index < W.shape[3]
                                ):
                                    dX[b, c, k, g] += (
                                        out_grad[b, f, i, j] * W[f, c, k_index, g_index]
                                    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dX


if __name__ == "__main__":
    np.random.seed(1992)

    param = {
        "X": np.random.rand(5, 3, 6, 6),
        "W": np.random.rand(2, 3, 2, 2),
        "stride": (3, 3),
    }
    grad = np.ones((5, 2, 2, 2))
    dx = convolution2D_backward_input(grad, **param)

    w_t = torch.from_numpy(param["W"]).float()
    x_t = torch.from_numpy(param["X"]).float()
    x_t.requires_grad = True
    w_t.requires_grad = True
    c = torch.nn.functional.conv2d(x_t, w_t, stride=param["stride"], padding="valid")

    loss = c.sum()
    loss.backward()
    dx_t = x_t.grad.cpu().detach().numpy()

    assert dx.shape == dx_t.shape
    print("Error is :", np.sum((dx - dx_t) ** 2))
    print("dX_t is :", np.sum(dx_t))
    print("dX is :", np.sum(dx))
