"""
Purpose:
    Compute CSS syndromes for binary symplectic error vectors.

Process:
    Use Hx to detect Z components and Hz to detect X components, then return
    either split or concatenated syndrome data.

Theory link:
    Stabilizer measurements reveal parity violations without identifying the
    encoded logical state directly.
"""

from typing import Tuple

import numpy as np


def compute_css_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex: np.ndarray, ez: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For CSS codes:
        sX  Hx * ez^T  (X-checks detect Z-part)
        sZ  Hz * ex^T  (Z-checks detect X-part)

    Role in pipeline:
        Produces the two syndrome vectors decoded separately by the X and Z
        component decoders.
    """
    sX = (Hx @ ez) % 2
    sZ = (Hz @ ex) % 2
    return sX.astype(np.uint8), sZ.astype(np.uint8)


def compute_full_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """
    Concatenate X-check and Z-check syndromes into one vector.
    """
    sX, sZ = compute_css_syndrome(Hx, Hz, ex, ez)
    return np.concatenate([sX, sZ]).astype(np.uint8)


def batch_css_syndrome(Hx: np.ndarray, Hz: np.ndarray, ex_batch: np.ndarray, ez_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CSS syndromes for a batch of sampled errors.
    """
    sX = (ez_batch @ Hx.T) % 2
    sZ = (ex_batch @ Hz.T) % 2
    return sX.astype(np.uint8), sZ.astype(np.uint8)


def syndrome_matches(H: np.ndarray, estimated_error: np.ndarray, target_syndrome: np.ndarray) -> bool:
    """
    Check whether an estimated binary error satisfies a target syndrome.
    """
    return np.array_equal(((H @ estimated_error) % 2).astype(np.uint8), target_syndrome.astype(np.uint8))
