import numpy as np


def css_commutation_check(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    if Hx.ndim != 2 or Hz.ndim != 2:
        raise ValueError("Hx and Hz must be 2D arrays")
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns")
    return np.all((Hx @ Hz.T) % 2 == 0)


def build_full_symplectic_check(Hx: np.ndarray, Hz: np.ndarray) -> np.ndarray:
    mx, n = Hx.shape
    mz, n2 = Hz.shape
    if n != n2:
        raise ValueError("Hx and Hz must have the same number of columns")

    top = np.concatenate([Hx, np.zeros((mx, n), dtype=np.uint8)], axis=1)
    bottom = np.concatenate([np.zeros((mz, n), dtype=np.uint8), Hz], axis=1)
    return np.concatenate([top, bottom], axis=0).astype(np.uint8)