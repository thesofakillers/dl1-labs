import typing as tg
import numpy as np
import numpy.typing as npt
import torch


def convolution2D_backward_filter(
    out_grad: npt.NDArray, X: npt.NDArray, W: npt.NDArray, stride: tg.Tuple[int]
):
    """
    A implementation of the backward pass for a convolutional layer.

    Parameters
    ----------
    out_grad : npt.NDArray
        gradient of the Loss with respect to the output of the conv
        layer with shape (N, F, OW, OH)
    X : npt.NDArray
        input data of shape (N, C, IH, IW)
    W : npt.NDArray
        Filter weight of shape (F, C, FH, FW)
    stride : tuple of int
        a tuple of (sh, sw)

    Returns
    -------
    dW : npt.NDArray
        Gradient with respect to W
    """
    ###########################################################################
    # Implement the convolutional backward pass.                              #
    ###########################################################################
    dW = np.zeros_like(W, dtype=np.float64)
    for f in range(W.shape[0]):
        for c in range(W.shape[1]):
            for k in range(W.shape[2]):
                for g in range(W.shape[3]):
                    for b in range(out_grad.shape[0]):
                        for i in range(out_grad.shape[2]):
                            for j in range(out_grad.shape[3]):
                                dW[f, c, k, g] += (
                                    out_grad[b, f, i, j]
                                    * X[b, c, stride[0] * i + k, stride[1] * j + g]
                                )
    ##########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dW


if __name__ == "__main__":
    np.random.seed(1345)

    param = {
        "X": np.random.rand(2, 3, 10, 10),
        "W": np.random.rand(7, 3, 4, 4),
        "stride": (2, 2),
    }
    c_1 = np.ones((2, 7, 4, 4))
    dw = convolution2D_backward_filter(c_1, **param)
    w_t = torch.from_numpy(param["W"]).float()
    x_t = torch.from_numpy(param["X"]).float()
    x_t.requires_grad = True
    w_t.requires_grad = True
    c = torch.nn.functional.conv2d(x_t, w_t, stride=param["stride"], padding="valid")

    loss = c.sum()
    loss.backward()
    dw_t = w_t.grad.cpu().detach().numpy()

    print("Error  :", np.sum((dw - dw_t) ** 2))
    print("dW_t  :", np.sum(dw_t))
    print("dW  :", np.sum(dw))
