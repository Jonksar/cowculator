import numpy as np

def vectorScaleMSE(x, y):
    """
    Vector Scaling, Minimizing Square Error:
    Find constant c for y = c * x such that
    (y - c * x).^2 is minimized.

    Example:
    x = np.arange(100)
    y = x * 5 + np.random.random(x.shape)

    :param x, y:
     np.array, which gets flattened if not 1D.
    :return: real number c
    """

    y, x = y.reshape(-1), x.reshape(-1)

    return np.dot(y, x) / np.dot(x, x)

