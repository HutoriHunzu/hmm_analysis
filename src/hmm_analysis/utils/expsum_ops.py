import numpy as np
from numpy.typing import NDArray
from numba import jit

MINUS_INF = -2 ** 32

@jit(cache=True, nopython=True, fastmath=True)
def logsumexp_3d(m: NDArray):
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
def logsumexp_2d(m: NDArray):
    """
    e.g. a = [(1, 0), (1, 1)]
        logsumexp_2d(a) = [log(e + 1), log(2e)]

    :param m: 2D numpy array
    :return: logsumexp of each row in the 2D array
    """
    # handle empty array case - numba-safe checks
    if m.shape[0] == 0:
        # Return empty array with correct dtype
        return np.empty(0, dtype=np.float64)
    if m.shape[1] == 0:
        # Return array of -inf for each row when columns are empty
        return np.full(m.shape[0], -np.inf)
    
    # find maximum
    result = np.empty(m.shape[0])
    for i in range(m.shape[0]):
        result[i] = logsumexp_1d(m[i])
    return result


@jit(cache=True, nopython=True, fastmath=True)
def logsumexp_1d(m: NDArray):
    """
    e.g. a = [1, 0, 1, 2]
        logsumexp_1d(a) = log(e + 1 + e + e^2)
    :param m: 1D numpy array
    :return: logsumexp of 1D array
    """
    # handle empty array case - numba-safe check
    if len(m) == 0:
        return -np.inf
    
    # find maximum
    max_scalar = np.max(m)
    if max_scalar < MINUS_INF:
        return -np.inf
    
    # sub the max from the matrix
    m = m - max_scalar

    # calculate sum and return
    return max_scalar + np.log(np.sum(np.exp(m)))


@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_matrix_vector(m: NDArray, v: NDArray):
    return logsumexp_2d(m + v)

@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_vector_matrix(v: NDArray, m: NDArray):
    return logsumexp_2d(m.T + v)


@jit(cache=True, nopython=True, fastmath=True)
def logexpdot_matrix_matrix(a: NDArray, b: NDArray):
    res = np.zeros((a.shape[0], b.shape[1]))
    b_t = b.T
    for i in range(a.shape[0]):
        res[i] = (logexpdot_matrix_vector(b_t, a[i]))
    # return np.array(res)
    return res


# def logsumexp_1d(m: NDArray):
#     pass

# if __name__ == '__main__':
#     a = np.array([[(5, 0), (1, 1)], [(0, 3), (0, 0)]])
#     result = logsumexp_3d(a)
#     print(result)
