import numpy as np
import numpy.typing as npt
import torch


def pool_forward(X: npt.NDArray, f: int, s: int):
    """
    Implements the forward pass of the pooling layer

    Parameters
    ----------
    X : npt.NDArray
        numpy array of shape (N, C, IH, IW)
    f : int
        filter size in height and width dim
    s : int
        stride size in height and width dim

    Returns
    -------
    pool : npt.NDArray
        output of the pool layer, with shape (N, C, OH, OW) where OH and OW given by
           OH = 1 + int((IH - f)/s)
           OW = 1 + int((IW - f)/s)
    """
    pool = None
    ###########################################################################
    #  Implement the max pooling forward pass.                                #
    ###########################################################################
    OH = 1 + int((X.shape[2] - f) / s)
    OW = 1 + int((X.shape[3] - f) / s)
    pool = np.zeros((X.shape[0], X.shape[1], OH, OW), dtype=np.float64)
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            for i in range(OH):
                for j in range(OW):
                    pool[b, c, i, j] = np.max(
                        X[b, c, i * s : i * s + f, j * s : j * s + f]
                    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return pool


if __name__ == "__main__":
    np.random.seed(1975)
    x = np.random.rand(2, 3, 23, 23)

    hyper_param = {"f": 2, "s": 11}
    c = pool_forward(x, **hyper_param)

    pooling = torch.nn.MaxPool2d(2, 11)

    x_t = torch.from_numpy(x)
    x_t.requires_grad = True
    pool_t = pooling(x_t).cpu().detach().numpy()

    assert c.shape == pool_t.shape
    assert (c == pool_t).all()

    print("your implementation is correct")
    print("output shape :", c.shape)
    print("output :", c)
    print("Error :", np.sum((c - pool_t) ** 2))
