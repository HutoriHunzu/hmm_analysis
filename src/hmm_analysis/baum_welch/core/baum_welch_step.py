from hmm_analysis.baum_welch.variable_updates import update_variables_log
from hmm_analysis.forward_backward import get_forward_backward_likelihood_log
from hmm_analysis.baum_welch.estimations import estimate_hidden_transition_log
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def step(data: np.ndarray, transition_log: np.ndarray, emission_log: np.ndarray,
         initial_log: np.ndarray):

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
