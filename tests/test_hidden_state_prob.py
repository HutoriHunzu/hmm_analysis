from hmm_analysis.baum_welch.estimations.hidden_state_prob import calc_hidden_state_prob, calc_hidden_state_prob_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('forward', 'backward', 'hidden_state_prob')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_hidden_state_prob(arrange_data):
    forward, backward, expected_result = arrange_data

    # calculate backward
    result = calc_hidden_state_prob(forward, backward)

    # assert
    assert_result(expected_result, result)


def test_hidden_state_prob_logexp(arrange_data):
    forward, backward, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        forward, backward = list(map(np.log, [forward, backward]))
        result = calc_hidden_state_prob_log(forward, backward)

        # assert
        assert_result_log(expected_result, result)


