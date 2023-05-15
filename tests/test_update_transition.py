from hmm_analysis.baum_welch.variable_updates.update_transition import calc_updated_transition, \
    calc_updated_transition_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('hidden_transition', 'hidden_state_prob', 'updated_transition')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_update_transition(arrange_data):
    transition_prob, hidden_state_prob, expected_result = arrange_data

    # calculate backward
    result = calc_updated_transition(transition_prob, hidden_state_prob)

    # assert
    assert_result(expected_result, result)


def test_update_transition_logexp(arrange_data):
    transition_prob, hidden_state_prob, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        transition_prob, hidden_state_prob = list(map(np.log, [transition_prob, hidden_state_prob]))
        result = calc_updated_transition_log(transition_prob, hidden_state_prob)

        # assert
        assert_result_log(expected_result, result)



