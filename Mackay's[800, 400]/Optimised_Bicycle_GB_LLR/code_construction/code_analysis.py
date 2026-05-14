"""
Purpose:
    Summarize CSS matrix dimensions, ranks, density, and weights.

Process:
    Compute GF(2) ranks and simple sparsity statistics, then print them in a
    compact form for simulation logs.

Theory link:
    The estimate k = n - rank(Hx) - rank(Hz) is the CSS logical dimension
    before accounting for any additional dependencies.
"""

from dataclasses import dataclass

import numpy as np

from code_construction.circulant import density
from core.helpers import gf2_rank


@dataclass
class CodeStats:
    n: int
    mx: int
    mz: int
    rank_hx: int
    rank_hz: int
    k_estimate: int
    hx_density: float
    hz_density: float
    hx_row_weights: np.ndarray
    hz_row_weights: np.ndarray
    hx_col_weights: np.ndarray
    hz_col_weights: np.ndarray


def analyze_css_code(Hx: np.ndarray, Hz: np.ndarray) -> CodeStats:
    """
    Return code statistics for a CSS pair.
    """
    n = Hx.shape[1]
    mx = Hx.shape[0]
    mz = Hz.shape[0]

    rank_hx = gf2_rank(Hx)
    rank_hz = gf2_rank(Hz)
    k_estimate = n - rank_hx - rank_hz

    return CodeStats(
        n=n,
        mx=mx,
        mz=mz,
        rank_hx=rank_hx,
        rank_hz=rank_hz,
        k_estimate=k_estimate,
        hx_density=density(Hx),
        hz_density=density(Hz),
        hx_row_weights=np.sum(Hx, axis=1),
        hz_row_weights=np.sum(Hz, axis=1),
        hx_col_weights=np.sum(Hx, axis=0),
        hz_col_weights=np.sum(Hz, axis=0),
    )


def print_code_stats(Hx: np.ndarray, Hz: np.ndarray) -> None:
    """
    Print matrix statistics before a simulation run.
    """
    stats = analyze_css_code(Hx, Hz)
    print("\n=== CSS Code Stats ===")
    print(f"n            = {stats.n}")
    print(f"mx, mz       = {stats.mx}, {stats.mz}")
    print(f"rank(Hx)     = {stats.rank_hx}")
    print(f"rank(Hz)     = {stats.rank_hz}")
    print(f"k estimate   = {stats.k_estimate}")
    print(f"density(Hx)  = {stats.hx_density:.4f}")
    print(f"density(Hz)  = {stats.hz_density:.4f}")
    print(f"Hx row wt    = min {stats.hx_row_weights.min()} | max {stats.hx_row_weights.max()}")
    print(f"Hz row wt    = min {stats.hz_row_weights.min()} | max {stats.hz_row_weights.max()}")
    print(f"Hx col wt    = min {stats.hx_col_weights.min()} | max {stats.hx_col_weights.max()}")
    print(f"Hz col wt    = min {stats.hz_col_weights.min()} | max {stats.hz_col_weights.max()}")


def print_matrix(name: str, M: np.ndarray) -> None:
    """
    Print one binary matrix row by row.
    """
    print(f"\n{name} ({M.shape[0]} x {M.shape[1]}):")
    for row in M:
        print(" ".join(str(int(x)) for x in row))


def print_bicycle_matrices(C: np.ndarray, Hx: np.ndarray, Hz: np.ndarray) -> None:
    """
    Print the circulant block and CSS check matrices.
    """
    print("\n=== Bicycle Code Matrices ===")
    print_matrix("C", C)
    print_matrix("Hx", Hx)
    print_matrix("Hz", Hz)
