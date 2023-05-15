import numpy as np
from hmm_analysis.utils.expsum_ops import logsumexp_2d
from numba import jit


def calc_updated_transition(transition_prob: np.ndarray, state_prob: np.ndarray):
    numerator = np.sum(transition_prob, axis=0)
    denomenator = np.sum(state_prob[:-1], axis=0)
    return (numerator.T / denomenator).T


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_transition_log(transition_prob_log: np.ndarray, state_prob_log: np.ndarray):
    shape = transition_prob_log.shape

    numerator = logsumexp_2d(transition_prob_log.reshape(-1, shape[1] * shape[2]).T).reshape(shape[1], shape[2])
    denomenator = logsumexp_2d(state_prob_log[:-1].T)

    return numerator - denomenator[:, None]


def calc_updated_transition_logexp(transition_prob: np.ndarray, state_prob: np.ndarray):
    with np.errstate(divide="ignore"):

        transition_prob_log = np.log(transition_prob)
        state_prob_log = np.log(state_prob)

        result = calc_updated_transition_log(transition_prob_log, state_prob_log)

    return np.exp(result)