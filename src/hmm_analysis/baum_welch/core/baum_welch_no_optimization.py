import numpy as np
from hmm_analysis.baum_welch.variable_updates import update_variables
from hmm_analysis.baum_welch.forward_backward import get_forward_backward_likelihood
from hmm_analysis.baum_welch.estimations import estimate_hidden_transition
from hmm_analysis.utils.proximity import check_proximity


def step(data: np.ndarray, transition: np.ndarray, emission: np.ndarray,
         initial: np.ndarray):

    # calculate forward and backward & casting them into numpy arrays
    forward, backward, norm = get_forward_backward_likelihood(data, initial,
                                                              transition, emission)

    # calculate the temporary variables, hidden state prob and transition prob
    hidden_state_prob, transition_prob = estimate_hidden_transition(data, forward, backward,
                                                                    transition, emission, norm)

    # updated variables - transition, emission, and initial
    initial, transition, emission = update_variables(data, hidden_state_prob,
                                                     transition_prob, emission)

    return transition, emission, initial, norm


def baum_welch(data: np.ndarray, transition: np.ndarray, emission: np.ndarray,
               initial: np.ndarray, niters: int, convergence_tol: float = 0.05):

    prev_likelihood = None

    for _ in range(niters):

        transition, emission, initial, likelihood = step(data, transition,
                                                         emission, initial)

        # checking convergence
        if check_proximity(prev_likelihood, likelihood, convergence_tol):
            break

        prev_likelihood = likelihood

    return transition, emission, initial

