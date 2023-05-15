from typing import Optional


def check_proximity(a: float, b: float, tol: Optional[float]):
    """Used to calculate convergence"""
    if not (a and b and tol):
        return False
    return abs(a - b) / a <= tol
