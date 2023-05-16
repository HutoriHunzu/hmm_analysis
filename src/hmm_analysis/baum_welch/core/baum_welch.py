import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List
from hmm_analysis.utils.proximity import check_proximity
from hmm_analysis.utils.casting import cast_log
from .baum_welch_step import step
from .baum_welch_result import list_of_log_estimations_to_bw_results, BaumWelchResult


def baum_welch(data: np.ndarray, transition: np.ndarray, emission: np.ndarray,
               initial: np.ndarray, niters: int, convergence_tol: Optional[float] = None,
               keep_all_results: bool = False) -> Union[List[BaumWelchResult], BaumWelchResult]:

    # casting all parameters to log
    transition, emission, initial = cast_log(transition, emission, initial)

    prev_likelihood_log = None

    # keeping all results
    results = []

    for i in tqdm(range(niters)):

        # these are log version of transition, emission and initial
        transition, emission, initial, likelihood_log = step(data, transition,
                                                             emission, initial)

        # updating data
        results.append((transition, emission, initial, likelihood_log))

        # checking convergence
        if check_proximity(prev_likelihood_log, likelihood_log, convergence_tol):
            print(f'Achieved convergence after: {i}/{niters} iterations')
            break

        prev_likelihood_log = likelihood_log

    results = list_of_log_estimations_to_bw_results(results)

    if keep_all_results:
        return results
    return results[-1]

