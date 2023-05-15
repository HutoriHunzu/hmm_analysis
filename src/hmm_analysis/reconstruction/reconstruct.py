from hmm_analysis.forward_backward import get_forward_backward_likelihood_log
from hmm_analysis.baum_welch.estimations.hidden_state_prob import calc_hidden_state_prob_log
from hmm_analysis.utils.casting import cast_log
import numpy as np


def reconstruct(data: np.ndarray, transition: np.ndarray, emission: np.ndarray,
                initial: np.ndarray):

    # casting parameters to log
    transition, emission, initial = cast_log(transition, emission, initial)

    forward_log, backward_log, likelihood = get_forward_backward_likelihood_log(data, initial,
                                                                                transition, emission)

    # the reconstructing is the maximum of the hidden state probability at each time
    hidden_state_prob = calc_hidden_state_prob_log(forward_log, backward_log, likelihood)

    # hidden state index
    return np.argmax(hidden_state_prob, axis=1)
