"""
Purpose:
    Utility functions for sparse circulant matrices.

Process:
    Convert supports into first rows, generate random sparse rows, and roll
    those rows into full circulant blocks.

Theory link:
    Bicycle codes use circulant matrices so the parity-check structure is
    sparse, regular, and easy to scale.
"""

from typing import Iterable, List, Optional

import numpy as np


def first_row_from_support(m: int, support: Iterable[int]) -> np.ndarray:
    """
    Create a binary first row from support indices modulo m.
    """
    row = np.zeros(m, dtype=np.uint8)
    for idx in support:
        row[idx % m] = 1
    return row


def random_sparse_first_row(m: int, weight: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate one random binary first row with the requested Hamming weight.
    """
    if weight < 0 or weight > m:
        raise ValueError("weight must satisfy 0 <= weight <= m")
    if rng is None:
        rng = np.random.default_rng()
    support = rng.choice(m, size=weight, replace=False)
    row = np.zeros(m, dtype=np.uint8)
    row[support] = 1
    return row


def circulant_from_first_row(first_row: np.ndarray) -> np.ndarray:
    """
    Build a circulant matrix by cyclically shifting the first row.
    """
    first_row = np.asarray(first_row, dtype=np.uint8)
    if first_row.ndim != 1:
        raise ValueError("first_row must be 1D")
    m = first_row.size
    C = np.zeros((m, m), dtype=np.uint8)
    for i in range(m):
        C[i] = np.roll(first_row, i)
    return C


def column_weights(H: np.ndarray) -> np.ndarray:
    return np.sum(H, axis=0)


def row_weights(H: np.ndarray) -> np.ndarray:
    return np.sum(H, axis=1)


def density(H: np.ndarray) -> float:
    return float(np.mean(H))
