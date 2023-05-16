import numpy as np
from .likelihood import likelihood_log, likelihood
from .backward import calc_backward_log, calc_backward
from .forward import calc_forward_log, calc_forward
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def get_forward_backward_likelihood_log(data, initial_log, transition_log, emission_log):
    forwards_log = calc_forward_log(data, transition_log, emission_log, initial_log)
    backwards_log = calc_backward_log(data, transition_log, emission_log)

    # calculating the likelihood, will serve as norm and as convergence parameter
    norm = likelihood_log(forwards_log, backwards_log)

    return forwards_log, backwards_log, norm


def get_forward_backward_likelihood(data, initial, transition, emission):
    forwards = calc_forward(data, transition, emission, initial)
    backwards = calc_backward(data, transition, emission)
    forwards, backwards = np.array(forwards), np.array(backwards)

    # calculating the likelihood, will serve as norm and as convergence parameter
    norm = likelihood(forwards, backwards)

    return forwards, backwards, norm

