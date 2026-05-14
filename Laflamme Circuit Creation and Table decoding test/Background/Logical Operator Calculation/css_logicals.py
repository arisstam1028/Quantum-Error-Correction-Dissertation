from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


class GF2:
    @staticmethod
    def as_u8(a) -> np.ndarray:
        return (np.asarray(a, dtype=np.uint8) & 1)

    @staticmethod
    def add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (GF2.as_u8(A) ^ GF2.as_u8(B)).astype(np.uint8)

    @staticmethod
    def dot(u: np.ndarray, v: np.ndarray) -> int:
        u = GF2.as_u8(u).ravel()
        v = GF2.as_u8(v).ravel()
        return int((np.bitwise_and(u, v).sum() & 1))

    @staticmethod
    def symplectic_product(p: np.ndarray, q: np.ndarray, n: int) -> int:
        """
        p=(x|z), q=(x'|z') -> x·z' + z·x' (mod 2)
        """
        p = GF2.as_u8(p).ravel()
        q = GF2.as_u8(q).ravel()
        x, z = p[:n], p[n:]
        xp, zp = q[:n], q[n:]
        return (GF2.dot(x, zp) ^ GF2.dot(z, xp)) & 1


class GF2Rank:
    """
    Rank + independent row extraction over GF(2).
    """

    @staticmethod
    def independent_rows(A: np.ndarray) -> Tuple[np.ndarray, List[int], int]:
        """
        Return (A_ind, kept_row_indices, rank).
        A_ind is a subset of A's rows that are linearly independent.
        """
        R = GF2.as_u8(A.copy())
        m, n = R.shape
        row_map = list(range(m))
        rank = 0
        row = 0

        for col in range(n):
            pivot = None
            for r in range(row, m):
                if R[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue

            if pivot != row:
                R[[row, pivot], :] = R[[pivot, row], :]
                row_map[row], row_map[pivot] = row_map[pivot], row_map[row]

            for r2 in range(m):
                if r2 != row and R[r2, col] == 1:
                    R[r2, :] ^= R[row, :]

            rank += 1
            row += 1
            if row == m:
                break

        kept = sorted(row_map[:rank])
        return GF2.as_u8(A[kept, :]), kept, rank

    @staticmethod
    def rank(A: np.ndarray) -> int:
        _, _, r = GF2Rank.independent_rows(A)
        return r


class GF2LinAlg:
    @staticmethod
    def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Return (RREF(A), pivot_cols) over GF(2).
        """
        R = GF2.as_u8(A.copy())
        m, n = R.shape
        pivots: List[int] = []
        row = 0

        for col in range(n):
            pivot = None
            for r in range(row, m):
                if R[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue

            if pivot != row:
                R[[row, pivot], :] = R[[pivot, row], :]

            for r2 in range(m):
                if r2 != row and R[r2, col] == 1:
                    R[r2, :] ^= R[row, :]

            pivots.append(col)
            row += 1
            if row == m:
                break

        return R, pivots

    @staticmethod
    def nullspace_basis(A: np.ndarray) -> np.ndarray:
        """
        Return a basis for Null(A) over GF(2).
        Output is a matrix N whose rows are basis vectors v with A v^T = 0.
        """
        A = GF2.as_u8(A)
        m, n = A.shape
        R, piv = GF2LinAlg.rref(A)
        piv_set = set(piv)
        free_cols = [c for c in range(n) if c not in piv_set]

        basis = []
        for f in free_cols:
            v = np.zeros(n, dtype=np.uint8)
            v[f] = 1

            # back-substitute using RREF structure
            for row_i, pcol in enumerate(piv):
                s = 0
                for j in free_cols:
                    if R[row_i, j] and v[j]:
                        s ^= 1
                v[pcol] = s

            basis.append(v)

        if not basis:
            return np.zeros((0, n), dtype=np.uint8)
        return np.vstack(basis)

    @staticmethod
    def is_in_rowspace(v: np.ndarray, B: np.ndarray) -> bool:
        """
        Test if v is in RowSpace(B) over GF(2).
        Equivalent: rank(B) == rank([B; v])
        """
        v = GF2.as_u8(v).reshape(1, -1)
        B = GF2.as_u8(B)
        r1 = GF2Rank.rank(B)
        r2 = GF2Rank.rank(np.vstack([B, v]))
        return r2 == r1

    @staticmethod
    def invert_kxk(M: np.ndarray) -> np.ndarray:
        """
        Invert a kxk matrix over GF(2). Raises ValueError if singular.
        """
        M = GF2.as_u8(M)
        k = M.shape[0]
        if M.shape != (k, k):
            raise ValueError("invert_kxk expects a square matrix.")
        aug = np.hstack([M, np.eye(k, dtype=np.uint8)])
        R, piv = GF2LinAlg.rref(aug)
        if len(piv) != k:
            raise ValueError("Matrix is singular over GF(2), cannot invert.")
        inv = R[:, k:]
        return GF2.as_u8(inv)


@dataclass
class CSSLogicalResult:
    n: int
    k: int
    Hx: np.ndarray
    Hz: np.ndarray
    Xbars: List[np.ndarray]   # each is length 2n [x|z]
    Zbars: List[np.ndarray]   # each is length 2n [x|z]


class CSSLogicalOperatorCalculator:
    """
    Compute logical operators for a CSS stabilizer code.

    Supports:
      - compute_from_HxHz(Hx, Hz)
      - compute_from_Hq_css(Hq)

    Returned logicals satisfy:
      - each Xbar commutes with all stabilizers
      - each Zbar commutes with all stabilizers
      - Xbar_i anticommutes with Zbar_i (paired)
    """

    @staticmethod
    def validate_css(Hx: np.ndarray, Hz: np.ndarray) -> None:
        Hx = GF2.as_u8(Hx)
        Hz = GF2.as_u8(Hz)
        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError("Hx and Hz must have the same number of columns.")
        prod = (Hx @ Hz.T) & 1
        if np.any(prod):
            bad = np.argwhere(prod == 1)
            raise ValueError(
                "CSS condition violated: Hx @ Hz.T != 0 (mod 2). "
                f"First few violations (rowHx,rowHz): {bad[:10].tolist()}"
            )

    @staticmethod
    def split_Hq_css(Hq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given Hq in [X|Z], verify it's CSS (each row purely X-type or purely Z-type),
        and return (Hx, Hz).
        """
        Hq = GF2.as_u8(Hq)
        if Hq.shape[1] % 2 != 0:
            raise ValueError("Hq must have 2n columns.")
        n = Hq.shape[1] // 2
        X = Hq[:, :n]
        Z = Hq[:, n:]

        x_rows = np.all(Z == 0, axis=1)
        z_rows = np.all(X == 0, axis=1)

        if not np.all(x_rows | z_rows):
            raise ValueError("Hq is not CSS: some rows have both X and Z parts.")

        Hx = X[x_rows, :]
        Hz = Z[z_rows, :]
        return Hx, Hz

    @staticmethod
    def compute_from_HxHz(Hx, Hz, *, reduce_dependent: bool = True) -> CSSLogicalResult:
        Hx = GF2.as_u8(Hx)
        Hz = GF2.as_u8(Hz)
        n = Hx.shape[1]
        if Hz.shape[1] != n:
            raise ValueError("Hx and Hz must have the same number of columns.")

        # Optional: reduce dependent checks for numerical stability / correct rank
        if reduce_dependent:
            Hx, _, rx = GF2Rank.independent_rows(Hx)
            Hz, _, rz = GF2Rank.independent_rows(Hz)

        CSSLogicalOperatorCalculator.validate_css(Hx, Hz)

        # Candidate bases
        NX = GF2LinAlg.nullspace_basis(Hz)  # x-vectors s.t. Hz x^T = 0
        NZ = GF2LinAlg.nullspace_basis(Hx)  # z-vectors s.t. Hx z^T = 0

        # Stabilizer rowspaces for quotienting
        Rx = Hx  # rowspace(Hx)
        Rz = Hz  # rowspace(Hz)

        # Pick nontrivial representatives (not in stabilizer rowspace)
        Xcand = [v for v in NX if not GF2LinAlg.is_in_rowspace(v, Rx)]
        Zcand = [v for v in NZ if not GF2LinAlg.is_in_rowspace(v, Rz)]

        # Infer k from stabilizer ranks (for CSS): k = n - rank(Hx) - rank(Hz)
        k = n - GF2Rank.rank(Hx) - GF2Rank.rank(Hz)
        if k < 0:
            raise ValueError(f"Inferred k is negative (n={n}). Check matrices.")
        if k == 0:
            return CSSLogicalResult(n=n, k=0, Hx=Hx, Hz=Hz, Xbars=[], Zbars=[])

        if len(Xcand) < k or len(Zcand) < k:
            raise ValueError(
                f"Not enough independent logical candidates found. "
                f"Need k={k}, got Xcand={len(Xcand)}, Zcand={len(Zcand)}. "
                f"(This can happen if candidates chosen are not spanning the quotient.)"
            )

        # Take first k candidates (simple deterministic choice)
        Xlog = np.vstack(Xcand[:k])  # k x n
        Zlog = np.vstack(Zcand[:k])  # k x n

        # Pair them so X_i · Z_j = delta_ij
        M = (Xlog @ Zlog.T) & 1  # k x k
        Minv = GF2LinAlg.invert_kxk(M)
        Zlog_paired = (Minv @ Zlog) & 1

        # Package as [x|0] and [0|z]
        Xbars = [np.concatenate([Xlog[i], np.zeros(n, dtype=np.uint8)]) for i in range(k)]
        Zbars = [np.concatenate([np.zeros(n, dtype=np.uint8), Zlog_paired[i]]) for i in range(k)]

        return CSSLogicalResult(n=n, k=k, Hx=Hx, Hz=Hz, Xbars=Xbars, Zbars=Zbars)

    @staticmethod
    def compute_from_Hq_css(Hq, *, reduce_dependent: bool = True) -> CSSLogicalResult:
        Hq = GF2.as_u8(Hq)
        Hx, Hz = CSSLogicalOperatorCalculator.split_Hq_css(Hq)
        return CSSLogicalOperatorCalculator.compute_from_HxHz(Hx, Hz, reduce_dependent=reduce_dependent)