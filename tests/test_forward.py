from hmm_analysis.forward_backward import calc_forward, calc_forward_log
from .assert_with_error import assert_result, assert_result_log
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ("sequence", "transition", "emission", "initial", "forward")


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_forward(arrange_data):
    data, transition, emission, initial, expected_result = arrange_data

    # calculate backward
    result = calc_forward(data, transition, emission, initial)

    # assert
    assert_result(expected_result, result)


def test_forward_logexp(arrange_data):
    data, transition, emission, initial, expected_result = arrange_data

    # calculate backward
    with np.errstate(divide="ignore"):
        transition, emission, initial = list(
            map(np.log, [transition, emission, initial])
        )
        result = calc_forward_log(data, transition, emission, initial)

        # assert
        assert_result_log(expected_result, result)
