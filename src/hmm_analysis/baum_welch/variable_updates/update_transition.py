import numpy as np
from hmm_analysis.utils.expsum_ops import logsumexp_2d, logsumexp_3d
from numba import jit


def calc_updated_transition(transition_prob: np.ndarray, state_prob: np.ndarray):
    numerator = np.sum(transition_prob, axis=0)
    denomenator = np.sum(state_prob[:-1], axis=0)
    return (numerator.T / denomenator).T


# @jit(nopython=True, fastmath=True, cache=True)
# def calc_updated_transition_log(transition_prob_log: np.ndarray, state_prob_log: np.ndarray):
#     shape = transition_prob_log.shape
#
#     numerator = logsumexp_2d(transition_prob_log.reshape(-1, shape[1] * shape[2]).T).reshape(shape[1], shape[2])
#     denominator = logsumexp_2d(state_prob_log[:-1].T)
#
#     return numerator - denominator[:, None]

@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_transition_log(transition_prob_log: np.ndarray, state_prob_log: np.ndarray):

    numerator, denominator = _calc_updated_transition_log_numerator_denominator(transition_prob_log,
                                                                                state_prob_log)

    return numerator - denominator


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_transition_log_multi_sequence(transition_prob_log_lst: np.ndarray,
                                               state_prob_log_lst: np.ndarray):

    numerators, denominators = np.empty((len(transition_prob_log_lst), transition_prob_log_lst[0].shape[1],
                                         transition_prob_log_lst[0].shape[2])), \
                               np.empty((len(transition_prob_log_lst), state_prob_log_lst[0].shape[1], 1))

    # for transition_prob_log, state_prob_log in zip(transition_prob_log_lst, state_prob_log_lst):
    for i in range(len(transition_prob_log_lst)):
        transition_prob_log = transition_prob_log_lst[i]
        state_prob_log = state_prob_log_lst[i]

        numerator, denominator = _calc_updated_transition_log_numerator_denominator(transition_prob_log,
                                                                                    state_prob_log)
        numerators[i] = numerator
        denominators[i] = denominator

    sum_numerator = logsumexp_3d(numerators)
    sum_denominator = logsumexp_3d(denominators)

    return sum_numerator - sum_denominator


@jit(nopython=True, fastmath=True, cache=True)
def _calc_updated_transition_log_numerator_denominator(transition_prob_log: np.ndarray,
                                                       state_prob_log: np.ndarray):
    shape = transition_prob_log.shape

    numerator = logsumexp_2d(transition_prob_log.reshape(-1, shape[1] * shape[2]).T).reshape(shape[1], shape[2])
    denominator = logsumexp_2d(state_prob_log[:-1].T)[:, None]
    return numerator, denominator


def calc_updated_transition_logexp(transition_prob: np.ndarray, state_prob: np.ndarray):
    with np.errstate(divide="ignore"):

        transition_prob_log = np.log(transition_prob)
        state_prob_log = np.log(state_prob)

        result = calc_updated_transition_log(transition_prob_log, state_prob_log)

    return np.exp(result)