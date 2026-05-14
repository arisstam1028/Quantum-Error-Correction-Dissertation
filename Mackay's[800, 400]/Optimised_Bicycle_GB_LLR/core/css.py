"""
Purpose:
    CSS-code helpers for commutation and symplectic check construction.

Process:
    Validate Hx and Hz dimensions, test the CSS commutation condition, and
    optionally combine the two check matrices into one symplectic form.

Theory link:
    CSS stabilizers commute exactly when Hx Hz transpose is zero over GF(2).
"""

import numpy as np


def css_commutation_check(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    """
    Return True when X and Z stabilizer checks commute.
    """
    if Hx.ndim != 2 or Hz.ndim != 2:
        raise ValueError("Hx and Hz must be 2D arrays")
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns")
    return np.all((Hx @ Hz.T) % 2 == 0)


def build_full_symplectic_check(Hx: np.ndarray, Hz: np.ndarray) -> np.ndarray:
    """
    Build a block symplectic check matrix from CSS components.
    """
    mx, n = Hx.shape
    mz, n2 = Hz.shape
    if n != n2:
        raise ValueError("Hx and Hz must have the same number of columns")

    top = np.concatenate([Hx, np.zeros((mx, n), dtype=np.uint8)], axis=1)
    bottom = np.concatenate([np.zeros((mz, n), dtype=np.uint8), Hz], axis=1)
    return np.concatenate([top, bottom], axis=0).astype(np.uint8)
