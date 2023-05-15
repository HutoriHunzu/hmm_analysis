from hmm_analysis import baum_welch
from .assert_with_error import assert_result
from .loader import generate_filtered_data
import numpy as np
import pytest


KEYS = ('sequence', 'transition', 'emission', 'initial', 'updated_transition', 'updated_emission')


@pytest.fixture(params=generate_filtered_data(set(KEYS)))
def arrange_data(request):
    return [np.array(request.param[k]) for k in KEYS]


def test_update_baum_welch(arrange_data):
    data, transition, emission, initial, expected_transition, expected_emission = arrange_data

    # calculate backward
    updated_transition, updated_emission, updated_initial = baum_welch(data, transition, emission, initial, niters=1)

    # assert
    assert_result(expected_transition, updated_transition)
    assert_result(expected_emission, updated_emission)

