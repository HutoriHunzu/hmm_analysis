import numpy as np


def cast_log(*args):
    return [np.log(elem) for elem in args]


def cast_exp(*args):
    return [np.exp(elem) for elem in args]
