from __future__ import annotations


def check_proximity(a: float, b: float, tol: float | None):
    """Used to calculate convergence"""
    if not (a and b and tol):
        return False
    return abs(a - b) / a <= tol
