from __future__ import annotations

from hmm_analysis.baum_welch.variable_updates import (
    update_variables_log,
    update_variables_log_multi_sequence,
)
from hmm_analysis.forward_backward import get_forward_backward_likelihood_log
from hmm_analysis.baum_welch.estimations import estimate_hidden_transition_log
from hmm_analysis.utils.casting import cast_log
from numpy.typing import NDArray
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def step(
    data: NDArray, transition_log: NDArray, emission_log: NDArray, initial_log: NDArray
):
    """Single sequence Baum-Welch step implementation."""
    # calculate forward and backward & casting them into numpy arrays
    forward_log, backward_log, norm = get_forward_backward_likelihood_log(
        data, initial_log, transition_log, emission_log
    )

    # calculate the temporary variables, hidden state prob and transition prob
    hidden_state_prob_log, transition_prob_log = estimate_hidden_transition_log(
        data, forward_log, backward_log, transition_log, emission_log, norm
    )

    # updated variables - transition, emission, and initial
    initial_log, transition_log, emission_log = update_variables_log(
        data, hidden_state_prob_log, transition_prob_log, emission_log
    )

    return transition_log, emission_log, initial_log, norm


def baum_welch_iter(
    data: NDArray | list[NDArray],
    transition: NDArray,
    emission: NDArray,
    initial: NDArray,
    multi_sequence: bool = False,
):
    """Infinite iterator for Baum-Welch algorithm that yields results per iteration.

    This is a clean, infinite iterator with no stopping criteria - users have full control
    over when to stop based on convergence, iteration count, time limits, or any custom criteria.
    User explicitly controls single vs multi-sequence processing.

    Args:
        data: Observation sequences (single array or list of arrays for multi-sequence)
        transition: Initial transition matrix guess
        emission: Initial emission matrix guess
        initial: Initial probability vector guess
        multi_sequence: Whether to use multi-sequence processing (default False)

    Yields:
        tuple: (transition_log, emission_log, initial_log, likelihood_log) for each iteration
    """
    # casting all parameters to log space once
    transition_log, emission_log, initial_log = cast_log(transition, emission, initial)

    # infinite iterator - user controls stopping
    while True:
        # user explicitly controls single vs multi-sequence processing
        if multi_sequence:
            transition_log, emission_log, initial_log, likelihood_log = (
                step_multi_sequences(data, transition_log, emission_log, initial_log)
            )
        else:
            transition_log, emission_log, initial_log, likelihood_log = step(
                data, transition_log, emission_log, initial_log
            )

        yield transition_log, emission_log, initial_log, likelihood_log


@jit(nopython=True, fastmath=True, cache=True)
def step_multi_sequences(
    data: list[NDArray],
    transition_log: NDArray,
    emission_log: NDArray,
    initial_log: NDArray,
):
    """Multi-sequence Baum-Welch step implementation."""
    hidden_state_prob_log_lst, transition_prob_log_lst = [], []

    for sequence in data:
        # calculate forward and backward & casting them into numpy arrays
        forward_log, backward_log, norm = get_forward_backward_likelihood_log(
            sequence, initial_log, transition_log, emission_log
        )

        # calculate the temporary variables, hidden state prob and transition prob
        hidden_state_prob_log, transition_prob_log = estimate_hidden_transition_log(
            sequence, forward_log, backward_log, transition_log, emission_log, norm
        )

        hidden_state_prob_log_lst.append(hidden_state_prob_log)
        transition_prob_log_lst.append(transition_prob_log)

    # updated variables - transition, emission, and initial
    initial_log, transition_log, emission_log = update_variables_log_multi_sequence(
        data, hidden_state_prob_log_lst, transition_prob_log_lst, emission_log
    )

    return transition_log, emission_log, initial_log, norm
