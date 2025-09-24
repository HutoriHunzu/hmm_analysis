import numpy as np
from hmm_analysis.forward_backward import likelihood_log, likelihood
from numba import jit


def calc_hidden_state_prob(
    forward_lst: np.ndarray, backward_lst: np.ndarray, norm=None
):
    # this is the probability to the observables (shouldn't depend on t)
    if not norm:
        norm = likelihood(forward_lst, backward_lst)

    omega = forward_lst * backward_lst / norm

    return omega


@jit(cache=True, nopython=True, fastmath=True)
def calc_hidden_state_prob_log(
    forward_lst_log: np.ndarray, backward_lst_log: np.ndarray, norm=None
):
    # this is the probability to the observables (shouldn't depend on t)
    if not norm:
        norm = likelihood_log(forward_lst_log, backward_lst_log)

    omega_log = forward_lst_log + backward_lst_log - norm

    return omega_log


def calc_hidden_state_prob_logexp(forward_lst: np.ndarray, backward_lst: np.ndarray):
    # this is the probability to the observables (shouldn't depend on t)
    # data_prob = np.sum(forward_lst * backward_lst, axis=1)
    with np.errstate(divide="ignore"):
        forward_lst_log, backward_lst_log = np.log(forward_lst), np.log(backward_lst)
        omega_log = calc_hidden_state_prob_log(forward_lst_log, backward_lst_log)

    return np.exp(omega_log)
