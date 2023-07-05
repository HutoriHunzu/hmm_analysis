import numpy as np
from numba import jit


@jit(cache=True, nopython=True, fastmath=True)
def logsumexp_3d(m: np.ndarray):
    """
    e.g. a = [[(1, 0), (1, 1)], [(0, 1), (0, 0)]]
        logsumexp_2d(a) = [(log(e + 1), log(2)), (log(2), log(e + 1))]

    :param m: 3D numpy array
    :return: logsumexp of each matrix, such that we end up with 2D array
    """
    shape = m.shape
    reshaped = m.reshape(shape[0], shape[1] * shape[2]).T
    result = logsumexp_2d(reshaped)
    result_shaped = result.reshape(shape[1], shape[2])
    return result_shaped


@jit(cache=True, nopython=True, fastmath=True)
def logsumexp_2d(m: np.ndarray):
    """
    e.g. a = [(1, 0), (1, 1)]
        logsumexp_2d(a) = [log(e + 1), log(2e)]

    :param m: 2D numpy array
    :return: logsumexp of each row in the 2D array
    """
    # find maximum
    return np.array([logsumexp_1d(row) for row in m])


@jit(cache=True, nopython=True, fastmath=True)
def logsumexp_1d(m: np.ndarray):
    """
    e.g. a = [1, 0, 1, 2]
        logsumexp_1d(a) = log(e + 1 + e + e^2)
    :param m: 1D numpy array
    :return: logsumexp of 1D array
    """
    # find maximum
    max_scalar = np.max(m)
    if max_scalar == -np.inf:
        return -np.inf

    # sub the max from the matrix
    m = m - max_scalar

    # calculate sum and return
    return max_scalar + np.log(np.sum(np.exp(m)))


@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_matrix_vector(m: np.ndarray, v: np.ndarray):
    return logsumexp_2d(m + v)

@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_vector_matrix(v: np.ndarray, m: np.ndarray):
    return logsumexp_2d(m.T + v)


@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_matrix_matrix(a: np.ndarray, b: np.ndarray):
    res = np.zeros((a.shape[0], b.shape[1]))
    b_t = b.T
    for i in range(a.shape[0]):
        res[i] = (logexpdot_matrix_vector(b_t, a[i]))
    # return np.array(res)
    return res


# def logsumexp_1d(m: np.ndarray):
#     pass

# if __name__ == '__main__':
#     a = np.array([[(5, 0), (1, 1)], [(0, 3), (0, 0)]])
#     result = logsumexp_3d(a)
#     print(result)
