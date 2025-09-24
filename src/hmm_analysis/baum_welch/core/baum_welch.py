from __future__ import annotations

import itertools
from tqdm import tqdm
from numpy.typing import NDArray
from .step import baum_welch_iter as _baum_welch_iter
from .result import list_of_log_estimations_to_bw_results, BaumWelchResult


def baum_welch_iter(data: NDArray | list[NDArray], transition: NDArray, emission: NDArray,
                    initial: NDArray, multi_sequence: bool = False):
    """Infinite iterator for Baum-Welch algorithm that yields results per iteration.

    This is a clean, infinite iterator with no stopping criteria - users have full control
    over when to stop based on convergence, iteration count, time limits, or any custom criteria.
    User explicitly controls single vs multi-sequence processing.

    Args:
        data: Observation sequences (single array or list of arrays for multi-sequence)
        transition: Initial transition matrix guess
        emission: Initial emission matrix guess
        initial: Initial probability vector guess
        multi_sequence: Whether to use multi-sequence processing (default False)

    Yields:
        BaumWelchResult: Result object for each iteration with updated parameters
    """
    for transition_log, emission_log, initial_log, likelihood_log in _baum_welch_iter(
            data, transition, emission, initial, multi_sequence):
        # Convert back to regular space for result
        result_data = [(transition_log, emission_log, initial_log, likelihood_log)]
        result = list_of_log_estimations_to_bw_results(result_data)[0]
        yield result


def baum_welch(data: NDArray | list[NDArray], transition: NDArray, emission: NDArray,
               initial: NDArray, niters: int, tqdm_on: bool = True,
               multi_sequence: bool = False) -> BaumWelchResult:
    """Baum-Welch algorithm for Hidden Markov Model parameter estimation.

    Estimates HMM parameters (transition, emission, initial probabilities) using
    the Baum-Welch algorithm with log-space computations for numerical stability.
    User explicitly controls single vs multi-sequence processing.

    Args:
        data: Observation sequences (single array or list of arrays for multi-sequence)
        transition: Initial transition matrix guess (left multiplication: P(X_i) * T)
        emission: Initial emission matrix guess
        initial: Initial probability vector guess
        niters: Number of iterations to run
        tqdm_on: Whether to show progress bar (default True)
        multi_sequence: Whether to use multi-sequence processing (default False)

    Returns:
        BaumWelchResult: Final parameter estimates after niters iterations
    """
    # Create infinite iterator and limit to niters
    infinite_iterator = baum_welch_iter(data, transition, emission, initial, multi_sequence)
    limited_iterator = itertools.islice(infinite_iterator, niters)

    if tqdm_on:
        limited_iterator = tqdm(limited_iterator, total=niters, desc="Baum-Welch")

    # Run iterations and return final result
    final_result = None
    for result in limited_iterator:
        final_result = result

    return final_result

