from dataclasses import dataclass
from hmm_analysis.utils.casting import cast_exp
import numpy as np


@dataclass
class BaumWelchResult:

    transition: np.ndarray
    emission: np.ndarray
    initial: np.ndarray
    likelihood_log: float


def list_of_log_estimations_to_bw_results(list_of_log_estimations):
    def _helper():
        for elem in list_of_log_estimations:
            transition, emission, initial, likelihood_log = elem
            transition, emission, initial = cast_exp(transition, emission, initial)
            yield BaumWelchResult(
                transition=transition,
                emission=emission,
                initial=initial,
                likelihood_log=likelihood_log
            )
    return list(_helper())
