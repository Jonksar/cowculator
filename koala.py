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
    Check if R is a rotation matrix in R^2 or R^3

    Example:
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # rotation of 90 deg around z
    isRotationMatrix(R) -> True

    :param R:
     Rotation Matrix (3x3)
    :return:
     Bool, is rotation matrix?
    """
    Rt = np.transpose(R)
    shouldYeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldYeIdentity)
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

def Kabsch_algorithm(X, Y):
    """
    Based upon:
    1) https://en.wikipedia.org/wiki/Kabsch_algorith
    2) http://nghiaho.com/?page_id=671

    Find rotation R and translation t:
    find R, st for Y = R * X' + t,
    such (Y - [R * X' + t])^2 is minimized.

    Example:
        np.set_printoptions(precision=8, suppress=True)  # For convenience
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        t = np.array([0, 3, 0])

        X = np.random.rand(10, 3)
        Y = np.dot(X, R) + t

        # print the rotation and transform
        print Kabsch_algorithm(X, Y)[0], '\n'
        print Kabsch_algorithm(X, Y)[1]


    :param X, Y:
        arrays of corresponding point coordinates, (3 x N)
    :return:
        R: rotation matrix, (3 x 3)
        t: translation, (3,)

    """
    assert X.shape == Y.shape, "X and Y must have the same size"

    # Keeping dims allows to centre the points without hustling
    centroid_X = np.mean(X, axis=0, keepdims=True)
    centroid_Y = np.mean(Y, axis=0, keepdims=True)

    # centre the points
    XX = X - centroid_X
    YY = Y - centroid_Y

    H = np.dot(XX.T, YY)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(U, Vt)

    # special reflection case
    if np.linalg.det(R) < 0:
        print "Reflection detected"
        Vt[2, :] *= -1
        R = np.dot(Vt, U.T)

    # find translation
    t = -(np.dot(centroid_X, R) - centroid_Y)

    return R, t


def sscm(v):

    """
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
    scew-symmetric cross-product matrix
    :param v:
    :return:
    """
    assert np.squeeze(v).shape == (3,), "Vector must be in R^3"

    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def normalize(v):
    return v / np.linalg.norm(v)


def align_vectors(x, y):
    """
    Find rotation matrix in R^3 such that y = np.dot(x, R)

    :param x, y: 3x1 vector
    :return: Rotation matrix such that
             y == np.dot(x, R)
    """
    x = normalize(x)
    y = normalize(y)

    tmp = np.eye(3) + sscm(np.cross(x, y))
    tmp2 = (1 - np.dot(x, y)) / (np.linalg.norm(np.cross(x, y))**2)
    R = tmp + np.linalg.matrix_power(sscm(np.cross(x, y)), 2) * tmp2

    return R.T
