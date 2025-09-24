from __future__ import annotations

import warnings
from tqdm import tqdm
from numpy.typing import NDArray
from .step import baum_welch_iter as _baum_welch_iter
from .result import list_of_log_estimations_to_bw_results, BaumWelchResult


def baum_welch_iter(data: NDArray | list[NDArray], transition: NDArray, emission: NDArray,
                    initial: NDArray, niters: int, convergence_tol: float | None = None,
                    multi_sequence: bool = False):
    """Pure iterator for Baum-Welch algorithm that yields results per iteration.

    This is a clean iterator with no progress display - users can wrap it with
    their own progress indicators or consume it directly.

    Args:
        data: Observation sequences (single array or list of arrays for multi-sequence)
        transition: Initial transition matrix guess
        emission: Initial emission matrix guess
        initial: Initial probability vector guess
        niters: Maximum number of iterations
        convergence_tol: Convergence tolerance (None to disable)
        multi_sequence: Whether data contains multiple sequences

    Yields:
        BaumWelchResult: Result object for each iteration with updated parameters
    """
    for transition_log, emission_log, initial_log, likelihood_log in _baum_welch_iter(
            data, transition, emission, initial, niters, convergence_tol, multi_sequence):
        # Convert back to regular space for result
        result_data = [(transition_log, emission_log, initial_log, likelihood_log)]
        result = list_of_log_estimations_to_bw_results(result_data)[0]
        yield result


def baum_welch(data: NDArray | list[NDArray], transition: NDArray, emission: NDArray,
               initial: NDArray, niters: int, convergence_tol: float | None = None,
               keep_all_results: bool = False, multi_sequence: bool = False,
               tqdm_on: bool = True) -> list[BaumWelchResult] | BaumWelchResult:
    """Baum-Welch algorithm for Hidden Markov Model parameter estimation.

    Estimates HMM parameters (transition, emission, initial probabilities) using
    the Baum-Welch algorithm with log-space computations for numerical stability.

    Args:
        data: Observation sequences (single array or list of arrays for multi-sequence)
        transition: Initial transition matrix guess (left multiplication: P(X_i) * T)
        emission: Initial emission matrix guess
        initial: Initial probability vector guess
        niters: Maximum number of iterations
        convergence_tol: Convergence tolerance (None to disable)
        keep_all_results: DEPRECATED - use baum_welch_iter() for iteration access
        multi_sequence: Whether data contains multiple sequences
        tqdm_on: Whether to show progress bar (default True)

    Returns:
        BaumWelchResult: Final parameter estimates (or list if keep_all_results=True)
    """
    if keep_all_results:
        warnings.warn(
            "The 'keep_all_results' parameter is deprecated. "
            "Use 'baum_welch_iter()' to access intermediate results.",
            DeprecationWarning,
            stacklevel=2
        )
        # For backward compatibility, collect all results
        iterator = baum_welch_iter(data, transition, emission, initial, niters,
                                 convergence_tol, multi_sequence)
        if tqdm_on:
            iterator = tqdm(iterator, total=niters, desc="Baum-Welch")
        results = list(iterator)
        return results

    # Return only the final result
    iterator = baum_welch_iter(data, transition, emission, initial, niters,
                             convergence_tol, multi_sequence)
    if tqdm_on:
        iterator = tqdm(iterator, total=niters, desc="Baum-Welch")

    final_result = None
    for result in iterator:
        final_result = result

    return final_result

