import numpy as np
from numba import jit
from hmm_analysis.forward_backward.likelihood import likelihood_log, likelihood


def calc_transition_prob(data: np.ndarray, forward_lst: np.ndarray,
                         backward_lst: np.ndarray, transition: np.ndarray,
                         emission: np.ndarray, norm=None):

    # norm
    if not norm:
        norm = likelihood(forward_lst, backward_lst)

    # shift backwards lst: backwards(k) --> backwards(k+1)
    # shift forward lst to not include the last item
    shifted_forward = forward_lst[:-1]
    shifted_backward = backward_lst[1:]

    # calculate the emission times backward column vector
    r_vec = []
    for i, d in enumerate(data[1:]):
        m = emission.T[d]
        b = shifted_backward[i]
        r_vec.append(m * b)
    r_vec = np.array(r_vec)

    # outer product of the column vector and the row vector
    q_lst = np.array([np.outer(l, r) for l, r in zip(shifted_forward, r_vec)])
    print(f'{q_lst[0]=}')

    # element wise multiplication with transition with norm
    q_lst = np.array([elem * transition / norm for elem in q_lst])

    return q_lst


@jit(cache=True, nopython=True, fastmath=True)
def calc_transition_prob_log(data: np.ndarray, forward_lst_log: np.ndarray,
                             backward_lst_log: np.ndarray, transition_log: np.ndarray,
                             emission_log: np.ndarray, norm=None) -> np.ndarray:

    # norm
    if not norm:
        norm = likelihood_log(forward_lst_log, backward_lst_log)

    # shift backwards lst: backwards(k) --> backwards(k+1)
    # shift forward lst to not include the last item
    shifted_forward_log = forward_lst_log[:-1]
    shifted_backward_log = backward_lst_log[1:]

    # shape
    rows, cols = shifted_forward_log.shape

    # calculate the emission times backward column vector
    res = np.empty(shape=(rows, cols, cols))
    for i, d in enumerate(data[1:]):
        m = emission_log.T[d]
        b = shifted_backward_log[i]
        row_vec = m + b
        column_vec = shifted_forward_log

        # handle edge case where data is very short
        if i < len(column_vec):
            res[i] = column_vec[i].reshape(cols, -1) + row_vec + transition_log - norm
        else:
            res[i] = np.full((cols, cols), -np.inf)

    # return result
    return res


def calc_transition_prob_logexp(data: np.ndarray, forward_lst: np.ndarray,
                                backward_lst: np.ndarray, transition: np.ndarray,
                                emission: np.ndarray):

    # making log
    with np.errstate(divide="ignore"):
        forward_lst_log = np.log(forward_lst)
        backward_lst_log = np.log(backward_lst)
        transition_log = np.log(transition)
        emission_log = np.log(emission)

        result = calc_transition_prob_log(data, forward_lst_log, backward_lst_log,
                                          transition_log, emission_log)

    result = np.exp(result)
    return result

