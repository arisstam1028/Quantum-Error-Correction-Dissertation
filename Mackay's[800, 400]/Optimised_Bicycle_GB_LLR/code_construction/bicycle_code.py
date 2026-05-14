"""
Purpose:
    Construct bicycle CSS parity-check matrices from circulant blocks.

Process:
    Build a sparse circulant C, form H0 from C and its transpose, optionally
    prune rows, and use the result as both Hx and Hz.

Theory link:
    The MacKay bicycle construction is dual-containing when H H transpose is
    zero over GF(2), allowing the same matrix to define X and Z checks.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from code_construction.circulant import (
    circulant_from_first_row,
    first_row_from_support,
    random_sparse_first_row,
)
from core.css import css_commutation_check


@dataclass
class BicycleCode:
    C: np.ndarray
    Hx: np.ndarray
    Hz: np.ndarray
    first_row: np.ndarray


def build_bicycle_code(
    m: int,
    first_row_support: Optional[list[int]] = None,
    random_row_weight: Optional[int] = None,
    seed: Optional[int] = None,
    prune_rows: bool = False,
    prune_count_x: int = 0,
    prune_count_z: int = 0,
) -> BicycleCode:
    """
    Build one bicycle CSS code instance.

    Role in pipeline:
        Generates comparison matrices and family members used by the runner.
    """
    if first_row_support is None and random_row_weight is None:
        raise ValueError("Provide either first_row_support or random_row_weight")

    rng = np.random.default_rng(seed)

    if first_row_support is not None:
        first_row = first_row_from_support(m, first_row_support)
    else:
        first_row = random_sparse_first_row(m, random_row_weight, rng=rng)

    C = circulant_from_first_row(first_row)
    Ct = C.T.copy()

    H0 = np.concatenate([C, Ct], axis=1).astype(np.uint8)

    Hx = H0.copy()
    Hz = H0.copy()

    if prune_rows:
        if prune_count_x > 0:
            keep_x = np.ones(Hx.shape[0], dtype=bool)
            drop_x = rng.choice(Hx.shape[0], size=min(prune_count_x, Hx.shape[0]), replace=False)
            keep_x[drop_x] = False
            Hx = Hx[keep_x]

        if prune_count_z > 0:
            keep_z = np.ones(Hz.shape[0], dtype=bool)
            drop_z = rng.choice(Hz.shape[0], size=min(prune_count_z, Hz.shape[0]), replace=False)
            keep_z[drop_z] = False
            Hz = Hz[keep_z]

    if not css_commutation_check(Hx, Hz):
        raise ValueError("Constructed bicycle code does not satisfy Hx Hz^T = 0 over GF(2)")

    return BicycleCode(C=C, Hx=Hx, Hz=Hz, first_row=first_row)
