import numpy as np
import numpy.typing as npt


def pool_back(grad_pool: npt.NDArray, X: npt.NDArray, f: int, s: int) -> npt.NDArray:
    """
    Implements the backward pass of the pooling layer

    Parameters
    ----------
    grad_pool : npt.NDArray
        gradient of cost with respect to the output of the pooling layer
    X : npt.NDArray
        input to pooling layer , numpy array with shape (N, C, IH, IW)
    f : int
        filter size in height and width dim
    s : int
        stride size in height and width dim

    Returns
    -------
    dX_pool : npt.NDArray
        gradient of cost with respect to the input of the pooling layer,
        same shape as X
    """
    ###########################################################################
    # Implement the max pooling backward pass.                               #
    ###########################################################################
    dX_pool = np.zeros_like(X)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    OH = 1 + int((X.shape[2] - f) / s)
    OW = 1 + int((X.shape[3] - f) / s)
    for b in range(dX_pool.shape[0]):
        for c in range(dX_pool.shape[1]):
            for i in range(OH):
                for j in range(OW):
                    window = X[b, c, i * s : i * s + f, j * s : j * s + f]
                    max_mask = window == np.max(window).astype(np.float64)
                    dX_pool[b, c, i * s : i * s + f, j * s : j * s + f] += (
                        max_mask * grad_pool[b, c, i, j]
                    )
    return dX_pool


if __name__ == "__main__":
    np.random.seed(19)
    X = np.random.rand(2, 3, 10, 10)
    g = np.random.rand(2, 3, 9, 9)

    f = 2
    s = 1
    dX = pool_back(g, X, f, s)

    print("mean of dX :", np.mean(dX))
    print("dX[1,2,2:5,2:5] = ", dX[1, 2, 2:5, 2:5])
