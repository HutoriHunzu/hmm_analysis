from __future__ import annotations

import numpy as np
from numba import jit
from hmm_analysis.utils.expsum_ops import logsumexp_2d


def calc_updated_initial(state_prob: np.ndarray):
    return state_prob[0]


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_initial_log(state_prob_log: np.ndarray):
    return state_prob_log[0]


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_initial_log_multi_sequence(state_prob_log_lst: list[np.ndarray]):
    result = np.empty((len(state_prob_log_lst), state_prob_log_lst[0].shape[1]))
    for i in range(len(state_prob_log_lst)):
        result[i] = state_prob_log_lst[i][0]
    result = logsumexp_2d(result.T)
    # for elem in state_prob_log_lst:
    #     result += elem[0]
    return result - np.log(len(state_prob_log_lst))
    # return sum([elem[0] for elem in state_prob_log_lst]) / len(state_prob_log_lst)
