import numpy as np
from numpy.typing import NDArray
from hmm_analysis.utils.expsum_ops import logexpdot_matrix_vector
from numba import jit


def calc_backward(data: NDArray, transition: NDArray, emission: NDArray) -> NDArray:
    # initiate result list
    res = []

    # iterating over data and constructing b_i(k)
    prob = np.ones(transition.shape[0])
    res.append(prob)

    for d in data[1:][::-1]:
        prob = (emission.T[d] * transition) @ prob
        res.append(prob)

    return np.array(res[::-1])


@jit(nopython=True, fastmath=True, cache=True)
def calc_backward_log(
    data: NDArray, transition_log: NDArray, emission_log: NDArray
) -> NDArray:
    # iterating over data and constructing f_i(k)
    emission_log_transpose = emission_log.T
    log_prob = np.zeros(transition_log.shape[0])
    res = np.empty(shape=(len(data), len(log_prob)))
    res[0] = log_prob

    for i, d in enumerate(data[1:][::-1]):
        log_prob = logexpdot_matrix_vector(
            emission_log_transpose[d] + transition_log, log_prob
        )
        res[i + 1] = log_prob

    return res[::-1]


def calc_backward_logexp(
    data: NDArray, transition: NDArray, emission: NDArray
) -> NDArray:
    # initiate result list
    with np.errstate(divide="ignore"):
        emission_log = np.log(emission)
        transition_log = np.log(transition)

        res = calc_backward_log(data, transition_log, emission_log)

    return np.exp(res)
