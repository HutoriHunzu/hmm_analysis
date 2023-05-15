from hmm_analysis.baum_welch.variable_updates.update_emission import calc_updated_emission, calc_updated_emission_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('sequence', 'hidden_state_prob', 'emission_range', 'updated_emission')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_update_emission(arrange_data):
    data, hidden_state_prob, emission_range, expected_result = arrange_data

    # calculate backward
    result = calc_updated_emission(data, hidden_state_prob, emission_range)

    # assert
    assert_result(expected_result, result)


def test_update_emission_log(arrange_data):
    data, hidden_state_prob, emission_range, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        hidden_state_prob = np.log(hidden_state_prob)

        result = calc_updated_emission_log(data, hidden_state_prob, emission_range)

        # assert
        assert_result_log(expected_result, result)

