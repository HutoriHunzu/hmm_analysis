from hmm_analysis.baum_welch.estimations.transition_prob import calc_transition_prob, calc_transition_prob_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('sequence', 'forward', 'backward', 'transition', 'emission', 'hidden_transition')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_transition_prob(arrange_data):
    data, forward, backward, transition, emission, expected_result = arrange_data

    # calculate backward
    result = calc_transition_prob(data, forward, backward, transition, emission)

    # assert
    assert_result(expected_result, result)


def test_transition_prob_logexp(arrange_data):
    data, forward, backward, transition, emission, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        forward, backward, transition, emission = list(map(np.log, [forward, backward, transition, emission]))

        result = calc_transition_prob_log(data, forward, backward, transition, emission)

        # assert
        assert_result_log(expected_result, result)
