from .hidden_state_prob import calc_hidden_state_prob_log, calc_hidden_state_prob
from .transition_prob import calc_transition_prob_log, calc_transition_prob
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def estimate_hidden_transition_log(
    data, forward_log, backward_log, transition_log, emission_log, norm
):
    hidden_state_prob_log = calc_hidden_state_prob_log(
        forward_log, backward_log, norm=norm
    )

    transition_prob_log = calc_transition_prob_log(
        data, forward_log, backward_log, transition_log, emission_log, norm=norm
    )

    return hidden_state_prob_log, transition_prob_log


def estimate_hidden_transition(data, forward, backward, transition, emission, norm):
    hidden_state_prob = calc_hidden_state_prob(forward, backward, norm=norm)

    transition_prob = calc_transition_prob(
        data, forward, backward, transition, emission, norm=norm
    )
    transition_prob = np.array(transition_prob)

    return hidden_state_prob, transition_prob
