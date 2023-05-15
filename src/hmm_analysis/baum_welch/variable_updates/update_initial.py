import numpy as np
from numba import jit


def calc_updated_initial(state_prob: np.ndarray):
    return state_prob[0]


@jit(nopython=True, fastmath=True, cache=True)
def calc_updated_initial_log(state_prob_log: np.ndarray):
    return state_prob_log[0]

