from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


M = 12
N = 2 * M

ROW_WEIGHT = 3
MAX_PRUNE_X = 4
MAX_PRUNE_Z = 4
VERBOSE = True


def circulant_from_support(m: int, support: Iterable[int]) -> np.ndarray:
    first_row = np.zeros(m, dtype=np.uint8)
    for s in support:
        first_row[s % m] = 1

    C = np.zeros((m, m), dtype=np.uint8)
    for i in range(m):
        C[i] = np.roll(first_row, i)

    return C


def gf2_rank(A: np.ndarray) -> int:
    Mtx = np.array(A, dtype=np.uint8, copy=True) % 2
    rows, cols = Mtx.shape
    rank = 0
    pivot_col = 0

    for r in range(rows):
        while pivot_col < cols:
            pivot_rows = np.where(Mtx[r:, pivot_col] == 1)[0]
            if pivot_rows.size == 0:
                pivot_col += 1
                continue

            pivot = r + pivot_rows[0]
            if pivot != r:
                Mtx[[r, pivot]] = Mtx[[pivot, r]]

            for rr in range(rows):
                if rr != r and Mtx[rr, pivot_col] == 1:
                    Mtx[rr] ^= Mtx[r]

            rank += 1
            pivot_col += 1
            break

        if pivot_col >= cols:
            break

    return rank


def css_commutes(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    return np.all((Hx @ Hz.T) % 2 == 0)


def density(H: np.ndarray) -> float:
    return float(np.mean(H))


def prune_rows(H: np.ndarray, rows_to_drop: tuple[int, ...]) -> np.ndarray:
    if not rows_to_drop:
        return H.copy()

    keep = np.ones(H.shape[0], dtype=bool)
    keep[list(rows_to_drop)] = False
    return H[keep]


def build_bicycle_code(support: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = circulant_from_support(M, support)
    H0 = np.concatenate([C, C.T], axis=1).astype(np.uint8)
    Hx = H0.copy()
    Hz = H0.copy()
    return C, Hx, Hz


def code_summary(Hx: np.ndarray, Hz: np.ndarray) -> dict:
    rank_hx = gf2_rank(Hx)
    rank_hz = gf2_rank(Hz)
    k_estimate = Hx.shape[1] - rank_hx - rank_hz

    hx_row_weights = np.sum(Hx, axis=1)
    hz_row_weights = np.sum(Hz, axis=1)
    hx_col_weights = np.sum(Hx, axis=0)
    hz_col_weights = np.sum(Hz, axis=0)

    return {
        "n": Hx.shape[1],
        "mx": Hx.shape[0],
        "mz": Hz.shape[0],
        "rank_hx": rank_hx,
        "rank_hz": rank_hz,
        "k_estimate": k_estimate,
        "density_hx": density(Hx),
        "density_hz": density(Hz),
        "hx_row_w_min": int(hx_row_weights.min()),
        "hx_row_w_max": int(hx_row_weights.max()),
        "hz_row_w_min": int(hz_row_weights.min()),
        "hz_row_w_max": int(hz_row_weights.max()),
        "hx_col_w_min": int(hx_col_weights.min()),
        "hx_col_w_max": int(hx_col_weights.max()),
        "hz_col_w_min": int(hz_col_weights.min()),
        "hz_col_w_max": int(hz_col_weights.max()),
        "commutes": css_commutes(Hx, Hz),
    }


def is_suitable_qldpc(Hx: np.ndarray, Hz: np.ndarray) -> tuple[bool, dict]:
    info = code_summary(Hx, Hz)

    checks = {
        "commutes": info["commutes"],
        "positive_k": info["k_estimate"] > 0,
        "sparse_hx": info["density_hx"] <= 0.30,
        "sparse_hz": info["density_hz"] <= 0.30,
        "nonzero_hx_columns": info["hx_col_w_min"] > 0,
        "nonzero_hz_columns": info["hz_col_w_min"] > 0,
    }

    info["checks"] = checks
    return all(checks.values()), info


def format_matrix_py(name: str, Mtx: np.ndarray) -> str:
    lines = [f"{name} = np.array(["]
    for row in Mtx.tolist():
        lines.append("    [" + ", ".join(str(int(x)) for x in row) + "],")
    lines.append("], dtype=np.uint8)")
    return "\n".join(lines)


def export_fixed_matrix_file(
    C: np.ndarray,
    Hx: np.ndarray,
    Hz: np.ndarray,
    support: tuple[int, ...],
    rows_dropped_x: tuple[int, ...],
    rows_dropped_z: tuple[int, ...],
    info: dict,
) -> Path:
    output_path = Path(__file__).resolve().parent / "bicycle_24.py"

    text = f"""import numpy as np

SUPPORT = {tuple(int(x) for x in support)}
ROWS_DROPPED_X = {tuple(int(x) for x in rows_dropped_x)}
ROWS_DROPPED_Z = {tuple(int(x) for x in rows_dropped_z)}

N = {info["n"]}
MX = {info["mx"]}
MZ = {info["mz"]}
RANK_HX = {info["rank_hx"]}
RANK_HZ = {info["rank_hz"]}
K_ESTIMATE = {info["k_estimate"]}
COMMUTES = {info["commutes"]}

{format_matrix_py("C", C)}

{format_matrix_py("Hx", Hx)}

{format_matrix_py("Hz", Hz)}
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def search_for_code() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, dict]:
    row_indices = tuple(range(M))

    for support in combinations(row_indices, ROW_WEIGHT):
        C, Hx_base, Hz_base = build_bicycle_code(support)

        for prune_x in range(MAX_PRUNE_X + 1):
            for rows_x in combinations(range(Hx_base.shape[0]), prune_x):
                Hx = prune_rows(Hx_base, rows_x)

                for prune_z in range(MAX_PRUNE_Z + 1):
                    for rows_z in combinations(range(Hz_base.shape[0]), prune_z):
                        Hz = prune_rows(Hz_base, rows_z)

                        ok, info = is_suitable_qldpc(Hx, Hz)
                        if ok:
                            return support, rows_x, rows_z, C, Hx, Hz, info

    raise RuntimeError(
        "No suitable 24-qubit bicycle code found. "
        "Try increasing MAX_PRUNE_X / MAX_PRUNE_Z or changing ROW_WEIGHT."
    )


def print_summary(
    support: tuple[int, ...],
    rows_x: tuple[int, ...],
    rows_z: tuple[int, ...],
    info: dict,
) -> None:
    print('\n Selected Bicycle 24 Code ')
    print(f'support             {support}')
    print(f'rows dropped X      {rows_x}')
    print(f'rows dropped Z      {rows_z}')
    print(f"n                   {info['n']}")
    print(f"mx, mz              {info['mx']}, {info['mz']}")
    print(f"rank(Hx)            {info['rank_hx']}")
    print(f"rank(Hz)            {info['rank_hz']}")
    print(f"k estimate          {info['k_estimate']}")
    print(f"density(Hx)         {info['density_hx']:.4f}")
    print(f"density(Hz)         {info['density_hz']:.4f}")
    print(f"Hx row wt min/max   {info['hx_row_w_min']} / {info['hx_row_w_max']}")
    print(f"Hz row wt min/max   {info['hz_row_w_min']} / {info['hz_row_w_max']}")
    print(f"Hx col wt min/max   {info['hx_col_w_min']} / {info['hx_col_w_max']}")
    print(f"Hz col wt min/max   {info['hz_col_w_min']} / {info['hz_col_w_max']}")
    print(f"commutes            {info['commutes']}")
    print("\nChecks:")
    for key, value in info["checks"].items():
        print(f"  {key}: {value}")


def main() -> None:
    support, rows_x, rows_z, C, Hx, Hz, info = search_for_code()

    if VERBOSE:
        print_summary(support, rows_x, rows_z, info)

    output_path = export_fixed_matrix_file(
        C=C,
        Hx=Hx,
        Hz=Hz,
        support=support,
        rows_dropped_x=rows_x,
        rows_dropped_z=rows_z,
        info=info,
    )

    print(f"\nCreated fixed matrix file:\n{output_path}")


if __name__ == "__main__":
    main()