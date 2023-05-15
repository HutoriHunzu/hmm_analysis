import numpy as np
from hmm_analysis.utils.expsum_ops import logexpdot_vector_matrix
from numba import jit


def calc_forward(data: np.ndarray, transition: np.ndarray, emission: np.ndarray, initial: np.ndarray) -> np.ndarray:
    # initiate result list
    res = []

    # iterating over data and constructing f_i(k)
    prob = emission.T[data[0]] * initial
    res.append(prob)

    for d in data[1:]:
        prob = emission.T[d] * (prob @ transition)
        res.append(prob)

    return np.array(res)


@jit(cache=True, nopython=True, fastmath=True)
def calc_forward_log(data: np.ndarray, transition_log: np.ndarray,
                     emission_log: np.ndarray, initial_log: np.ndarray) -> np.ndarray:

    # initiate result list
    # with np.errstate(divide="ignore"):

    # iterating over data and constructing f_i(k)
    # res = []
    emission_log_transpose = emission_log.T
    log_prob = emission_log_transpose[data[0]] + initial_log
    # res = np.array((log_prob, ))
    # res = [log_prob]
    res = np.empty(shape=(len(data), len(log_prob)))
    res[0] = log_prob

    # for d in data[1:]:
    for i, d in enumerate(data[1:]):
        log_prob = emission_log_transpose[d] + logexpdot_vector_matrix(log_prob, transition_log)
        # res.append(log_prob)
        res[i + 1] = log_prob
        # res = np.concatenate(res, log_prob)

    return res


def calc_forward_logexp(data: np.ndarray, transition: np.ndarray,
                     emission: np.ndarray, initial: np.ndarray) -> np.ndarray:

    # initiate result list
    with np.errstate(divide="ignore"):

        emission_log = np.log(emission)
        transition_log = np.log(transition)
        initial_log = np.log(initial)

        # iterating over data and constructing f_i(k)
        res = calc_forward_log(data, transition_log, emission_log, initial_log)

    return np.exp(res)




