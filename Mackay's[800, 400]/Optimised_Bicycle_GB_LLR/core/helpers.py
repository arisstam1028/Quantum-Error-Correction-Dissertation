import numpy as np


def _gf2_row_echelon_with_pivots(A: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    M = np.array(A, dtype=np.uint8, copy=True) % 2
    rows, cols = M.shape
    rank = 0
    pivot_col = 0
    pivots: list[int] = []

    for r in range(rows):
        while pivot_col < cols:
            pivot_rows = np.where(M[r:, pivot_col] == 1)[0]
            if pivot_rows.size == 0:
                pivot_col += 1
                continue

            pivot = r + pivot_rows[0]
            if pivot != r:
                M[[r, pivot]] = M[[pivot, r]]

            for rr in range(rows):
                if rr != r and M[rr, pivot_col] == 1:
                    M[rr] ^= M[r]

            pivots.append(pivot_col)
            rank += 1
            pivot_col += 1
            break

        if pivot_col >= cols:
            break

    return M, rank, pivots


def gf2_row_echelon(A: np.ndarray) -> tuple[np.ndarray, int]:
    M, rank, _ = _gf2_row_echelon_with_pivots(A)
    return M, rank


def gf2_rank(A: np.ndarray) -> int:
    _, rank = gf2_row_echelon(A)
    return rank


def ensure_binary_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.uint8)
    if not np.all((A == 0) | (A == 1)):
        raise ValueError("Matrix must be binary")
    return A


def binary_vector_to_str(v: np.ndarray) -> str:
    return "".join(str(int(x)) for x in v.tolist())


class GF2RowSpaceChecker:
    def __init__(self, H: np.ndarray):
        H = np.asarray(H, dtype=np.uint8)
        echelon, rank, pivots = _gf2_row_echelon_with_pivots(H)
        self.basis = echelon[:rank].copy()
        self.rank = rank
        self.pivots = pivots

    def contains(self, v: np.ndarray) -> bool:
        reduced = np.asarray(v, dtype=np.uint8).copy().reshape(-1)
        if self.rank == 0:
            return bool(np.all(reduced == 0))

        for row_idx, pivot_col in enumerate(self.pivots):
            if reduced[pivot_col] == 1:
                reduced ^= self.basis[row_idx]

        return bool(np.all(reduced == 0))


def in_rowspace(v: np.ndarray, H: np.ndarray) -> bool:
    return GF2RowSpaceChecker(H).contains(v)
