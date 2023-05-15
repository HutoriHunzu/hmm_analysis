from hmm_analysis.forward_backward import calc_backward, calc_backward_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('sequence', 'transition', 'emission', 'backward')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_backward(arrange_data):
    data, transition, emission, expected_result = arrange_data

    # calculate backward
    result = calc_backward(data, transition, emission)

    # assert
    assert_result(expected_result, result)


def test_backward_logexp(arrange_data):
    data, transition, emission, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        transition, emission = list(map(np.log, [transition, emission]))
        result = calc_backward_log(data, transition, emission)

        # assert
        assert_result_log(expected_result, result)
