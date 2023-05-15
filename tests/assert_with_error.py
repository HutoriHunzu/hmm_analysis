import numpy as np


def assert_result(expected_result: np.ndarray, result: np.ndarray, error: float = 0.2):
    mask = expected_result.copy()
    mask[mask == 0] = 1
    sb = np.abs(result - expected_result) / mask
    assert(np.all(sb < error))


def assert_result_log(expected_result: np.ndarray, result: list, error: float = 0.2):
    result = np.exp(result)
    assert_result(expected_result, result, error)