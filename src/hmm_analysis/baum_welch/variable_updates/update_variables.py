from .update_initial import (
    calc_updated_initial_log,
    calc_updated_initial,
    calc_updated_initial_log_multi_sequence,
)
from .update_transition import (
    calc_updated_transition_log,
    calc_updated_transition,
    calc_updated_transition_log_multi_sequence,
)
from .update_emission import (
    calc_updated_emission_log,
    calc_updated_emission,
    calc_updated_emission_log_multi_sequence,
)
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def update_variables_log(
    data, hidden_state_prob_log, transition_prob_log, emission_log
):
    # updated variables - transition, emission, and initial
    initial_log = calc_updated_initial_log(hidden_state_prob_log)
    transition_log = calc_updated_transition_log(
        transition_prob_log, hidden_state_prob_log
    )
    emission_log = calc_updated_emission_log(
        data, hidden_state_prob_log, emission_log.shape
    )
    return initial_log, transition_log, emission_log


@jit(nopython=True, fastmath=True, cache=True)
def update_variables_log_multi_sequence(
    data_lst, hidden_state_prob_log_lst, transition_prob_log_lst, emission_log
):
    # updated variables - transition, emission, and initial
    initial_log = calc_updated_initial_log_multi_sequence(hidden_state_prob_log_lst)
    transition_log = calc_updated_transition_log_multi_sequence(
        transition_prob_log_lst, hidden_state_prob_log_lst
    )
    emission_log = calc_updated_emission_log_multi_sequence(
        data_lst, hidden_state_prob_log_lst, emission_log.shape
    )
    return initial_log, transition_log, emission_log


def update_variables(data, hidden_state_prob, transition_prob, emission):
    # updated variables - transition, emission, and initial
    initial = calc_updated_initial(hidden_state_prob)
    transition = calc_updated_transition(transition_prob, hidden_state_prob)
    emission = calc_updated_emission(data, hidden_state_prob, emission.shape[1])
    return initial, transition, emission
