from hmm_analysis.baum_welch.variable_updates import update_variables_log, update_variables_log_multi_sequence
from hmm_analysis.forward_backward import get_forward_backward_likelihood_log
from hmm_analysis.baum_welch.estimations import estimate_hidden_transition_log
import numpy as np
from numpy.typing import NDArray
from numba import jit
from typing import List


@jit(nopython=True, fastmath=True, cache=True)
def step(data: NDArray, transition_log: NDArray, emission_log: NDArray,
         initial_log: NDArray):

    # calculate forward and backward & casting them into numpy arrays
    forward_log, backward_log, norm = get_forward_backward_likelihood_log(data, initial_log,
                                                                          transition_log, emission_log)

    # calculate the temporary variables, hidden state prob and transition prob
    hidden_state_prob_log, transition_prob_log = estimate_hidden_transition_log(data, forward_log, backward_log,
                                                                                transition_log, emission_log, norm)

    # updated variables - transition, emission, and initial
    initial_log, transition_log, emission_log = update_variables_log(data, hidden_state_prob_log,
                                                                     transition_prob_log, emission_log)

    return transition_log, emission_log, initial_log, norm


@jit(nopython=True, fastmath=True, cache=True)
def step_multi_sequences(data: List[NDArray], transition_log: NDArray, emission_log: NDArray,
                         initial_log: NDArray):

    hidden_state_prob_log_lst, transition_prob_log_lst = [], []

    for sequence in data:

        # calculate forward and backward & casting them into numpy arrays
        forward_log, backward_log, norm = get_forward_backward_likelihood_log(sequence, initial_log,
                                                                              transition_log, emission_log)

        # calculate the temporary variables, hidden state prob and transition prob
        hidden_state_prob_log, transition_prob_log = estimate_hidden_transition_log(sequence, forward_log, backward_log,
                                                                                    transition_log, emission_log, norm)

        hidden_state_prob_log_lst.append(hidden_state_prob_log)
        transition_prob_log_lst.append(transition_prob_log)

    # updated variables - transition, emission, and initial
    initial_log, transition_log, emission_log = update_variables_log_multi_sequence(data, hidden_state_prob_log_lst,
                                                                                    transition_prob_log_lst,
                                                                                    emission_log)

    return transition_log, emission_log, initial_log, norm
