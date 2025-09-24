from hmm_analysis.forward_backward import get_forward_backward_likelihood_log
from hmm_analysis.baum_welch.estimations.hidden_state_prob import calc_hidden_state_prob_log
from hmm_analysis.utils.casting import cast_log
import numpy as np
from numpy.typing import NDArray


def reconstruct(data: NDArray, transition: NDArray, emission: NDArray,
                initial: NDArray) -> NDArray:
    """Reconstruct hidden states using maximum likelihood estimation.

    Given HMM parameters and observations, estimates the most likely sequence
    of hidden states using the forward-backward algorithm.

    Args:
        data: Observation sequence
        transition: Transition matrix (left multiplication: P(X_i) * T)
        emission: Emission matrix
        initial: Initial probability vector

    Returns:
        Array of most likely hidden state indices for each observation
    """
    # casting parameters to log
    transition, emission, initial = cast_log(transition, emission, initial)

    forward_log, backward_log, likelihood = get_forward_backward_likelihood_log(data, initial,
                                                                                transition, emission)

    # the reconstructing is the maximum of the hidden state probability at each time
    hidden_state_prob = calc_hidden_state_prob_log(forward_log, backward_log, likelihood)

    # hidden state index
    return np.argmax(hidden_state_prob, axis=1)
