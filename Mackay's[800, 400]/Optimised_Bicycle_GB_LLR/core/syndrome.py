from typing import Tuple

import numpy as np


def compute_css_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex: np.ndarray, ez: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For CSS codes:
        sX = Hx * ez^T  (X-checks detect Z-part)
        sZ = Hz * ex^T  (Z-checks detect X-part)
    """
    sX = (Hx @ ez) % 2
    sZ = (Hz @ ex) % 2
    return sX.astype(np.uint8), sZ.astype(np.uint8)


def compute_full_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
    sX, sZ = compute_css_syndrome(Hx, Hz, ex, ez)
    return np.concatenate([sX, sZ]).astype(np.uint8)


def batch_css_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex_batch: np.ndarray, ez_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sX = (ez_batch @ Hx.T) % 2
    sZ = (ex_batch @ Hz.T) % 2
    return sX.astype(np.uint8), sZ.astype(np.uint8)


def syndrome_matches(H: np.ndarray, estimated_error: np.ndarray, target_syndrome: np.ndarray) -> bool:
    return np.array_equal(((H @ estimated_error) % 2).astype(np.uint8), target_syndrome.astype(np.uint8))