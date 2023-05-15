import numpy as np
from numba import jit
from hmm_analysis.utils.expsum_ops import logsumexp_1d


def likelihood(forward_lst, backward_lst):
    return np.sum(forward_lst * backward_lst, axis=1)[0]


@jit(cache=True, nopython=True, fastmath=True)
def likelihood_log(forward_lst_log, backward_lst_log):
    return logsumexp_1d(forward_lst_log[0] + backward_lst_log[0])
