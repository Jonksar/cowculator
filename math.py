import numpy as np
import math

def vectorScaleMSE(x, y):
    """
    Vector Scaling, Minimizing Square Error:
    Find constant c for y = c * x such that
    (y - c * x)^2 is minimized.

    Example:
    x = np.arange(100)
    y = x * 5 + np.random.random(x.shape)

    return value c is about 5

    :param x, y:
     np.array, which gets flattened if not 1D.
    :return: real number c
    """

    y, x = y.reshape(-1), x.reshape(-1)

    return np.dot(y, x) / np.dot(x, x)

def isRotationMatrix(R):
    """

    Example:
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    isRotationMatrix(R) -> True

    :param R:
     Rotation Matrix (3x3)
    :return:
     Bool, is rotation matrix?
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """
    :param R:
     Rotation matrix, (3x3)
    :return:
     Euler angles, (3,)
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def findRts(X, Y):
    """

    BUGGY!
    Find rotation R, translation t and scaling s:
    find R, s, t for Y = s * R * X' + t,
    such (Y - [s * R * X' + t])^2 is minimized.

    Based upon (AUTHOR: Anton Semechko):
    https://www.mathworks.com/matlabcentral/answers/40379-minimizing-mean-square-error-for-a-body-tracking-problem

    Example:

    x = np.array([[i, 0, 0] for i in range(20)])
    R = rotationMatrix(np.array([40, 10, 10]))
    y = np.dot(x, R) + np.random.random(x.shape)

    R_new, t, s = findRts(x, y)

    print rotationMatrixToEulerAngles(R_new)

    :param X, Y:
        arrays of corresponding point coordinates, (N x 3)
    :return:
        R: rotation matrix, (3 x 3)
        t: translation, (3,)
        s: scale, float

    """

    from numpy.linalg import svd

    assert False, "BUGGY?, are you sure you want to use it?"

    # Useful values
    m = Y.shape[0]

    # Centroids
    C_y = np.mean(Y, axis=0, keepdims=True)
    C_x = np.mean(X, axis=0, keepdims=True)

    # Center the point sets
    Y = Y - C_y
    X = X - C_x

    # compute the covariance matrix
    Cmat = np.dot(Y.T, X) / m

    # find rotation using SVD
    U, _, V = np.linalg.svd(Cmat)
    V = V.T
    V[:, [1, 2]] = V[:, [2, 1]]
    U[:, [1, 2]] = -U[:, [2, 1]]

    R = np.dot(np.dot(U, np.diag([1, 1, np.linalg.det(U * V.T)])), V.T)

    # Find the scaling factor
    Y_L2 = np.sum(Y * Y, axis=1)
    X_L2 = np.sum(X * X, axis=1)

    s = Y_L2 / X_L2
    s = np.mean(np.sqrt(s))

    # Translation
    t = (s * np.dot(R, C_x.T))

    # Reconstructing the Y from X and R, s, t
    Xnew = s * np.dot(R, X.T).T + t.reshape((1, -1))

    return R, t, s, Xnew