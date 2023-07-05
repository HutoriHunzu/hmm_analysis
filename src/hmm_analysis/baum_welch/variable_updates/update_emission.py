import numpy as np
from hmm_analysis.utils.expsum_ops import logsumexp_2d, logsumexp_3d
from numba import jit
from typing import Tuple


def calc_updated_emission(data: np.ndarray, state_prob: np.ndarray, emission_shape: int):
    denomenator = np.sum(state_prob, axis=0)
    res = []
    for i in range(emission_shape[1]):
        res.append(np.sum(state_prob[data == i], axis=0))
    numerator = np.array(res).T
    return (numerator.T / denomenator).T

#
# @jit(nopython=True, fastmath=True, cache=True)
# def calc_updated_emission_log(data: np.ndarray, state_prob_log: np.ndarray, emission_shape: Tuple[int, int]):
#
#     denomenator = logsumexp_2d(state_prob_log.T)
#
#     numerator = np.empty(shape=(emission_shape[1], emission_shape[0]))
#     for i in range(emission_shape[1]):
#         numerator[i] = logsumexp_2d(state_prob_log[data == i].T)
#
#     return numerator.T - denomenator[:, None]


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_emission_log(data: np.ndarray, state_prob_log: np.ndarray, emission_shape: Tuple[int, int]):

    numerator, denominator = _calc_updated_emission_log_numerator_denominator(data, state_prob_log, emission_shape)
    return numerator - denominator


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_emission_log_multi_sequence(data_lst: np.ndarray, state_prob_log_lst: np.ndarray,
                                             emission_shape: Tuple[int, int]):

    numerators, denominators = np.empty((len(data_lst), emission_shape[0], emission_shape[1])), \
                               np.empty((len(data_lst), state_prob_log_lst[0].shape[1], 1))

    for i in range(len(data_lst)):
    # for data, state_prob_log in zip(data_lst, state_prob_log_lst):
        data = data_lst[i]
        state_prob_log = state_prob_log_lst[i]
        numerator, denominator = _calc_updated_emission_log_numerator_denominator(data,
                                                                                  state_prob_log,
                                                                                  emission_shape)
        numerators[i] = numerator
        denominators[i] = denominator

    #
    sum_numerator = logsumexp_3d(numerators)
    sum_denominator = logsumexp_3d(denominators)

    return sum_numerator - sum_denominator


@jit(nopython=True, fastmath=True, cache=True)
def _calc_updated_emission_log_numerator_denominator(data: np.ndarray, state_prob_log: np.ndarray, emission_shape: Tuple[int, int]):

    denomenator = logsumexp_2d(state_prob_log.T)

    numerator = np.empty(shape=(emission_shape[1], emission_shape[0]))
    for i in range(emission_shape[1]):
        numerator[i] = logsumexp_2d(state_prob_log[data == i].T)

    return numerator.T, denomenator[:, None]


def calc_updated_emission_logexp(data: np.ndarray, state_prob: np.ndarray, emission_shape: int):

    with np.errstate(divide="ignore"):
        state_prob_log = np.log(state_prob)
        result = calc_updated_emission_log(data, state_prob_log, emission_shape)

    return np.exp(result)

