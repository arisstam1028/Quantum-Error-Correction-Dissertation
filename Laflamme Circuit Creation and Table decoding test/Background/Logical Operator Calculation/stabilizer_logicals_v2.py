from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple
import itertools
import numpy as np


# ============================================================
#  GF(2) utilities + symplectic product
# ============================================================

class GF2:
    @staticmethod
    def as_u8(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=np.uint8) & 1)

    @staticmethod
    def add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return GF2.as_u8(A ^ B)

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return GF2.as_u8((A @ B) & 1)

    @staticmethod
    def symplectic_inner(u: np.ndarray, v: np.ndarray, n: int) -> int:
        """⟨u,v⟩ = u_X·v_Z + u_Z·v_X (mod 2)"""
        u = GF2.as_u8(u).reshape(-1)
        v = GF2.as_u8(v).reshape(-1)
        ux, uz = u[:n], u[n:]
        vx, vz = v[:n], v[n:]
        return int((int(np.dot(ux, vz)) + int(np.dot(uz, vx))) & 1)


class GF2Rank:
    @staticmethod
    def rank(A: np.ndarray) -> int:
        M = GF2.as_u8(A.copy())
        m, n = M.shape
        r = 0
        c = 0
        while r < m and c < n:
            piv = None
            for i in range(r, m):
                if M[i, c] == 1:
                    piv = i
                    break
            if piv is None:
                c += 1
                continue
            if piv != r:
                M[[r, piv], :] = M[[piv, r], :]
            for i in range(m):
                if i != r and M[i, c] == 1:
                    M[i, :] ^= M[r, :]
            r += 1
            c += 1
        return r

    @staticmethod
    def independent_rows(A: np.ndarray) -> Tuple[np.ndarray, List[int], int]:
        """
        Returns a row-independent subset of A using GF(2) elimination.
        Output: (A_independent, kept_row_indices, rank)
        """
        M = GF2.as_u8(A.copy())
        m, n = M.shape
        kept: List[int] = []
        row = 0

        for col in range(n):
            pivot = None
            for p in range(row, m):
                if M[p, col] == 1:
                    pivot = p
                    break
            if pivot is None:
                continue

            if pivot != row:
                M[[row, pivot], :] = M[[pivot, row], :]

            for r2 in range(m):
                if r2 != row and M[r2, col] == 1:
                    M[r2, :] ^= M[row, :]

            kept.append(row)
            row += 1
            if row == m:
                break

        # The kept rows (after swaps) are at indices 0..rank-1 in the transformed matrix.
        # We want original indices; easiest is to redo with tracking:
        M = GF2.as_u8(A.copy())
        order = list(range(m))
        row = 0
        pivots = 0

        for col in range(n):
            pivot = None
            for p in range(row, m):
                if M[p, col] == 1:
                    pivot = p
                    break
            if pivot is None:
                continue

            if pivot != row:
                M[[row, pivot], :] = M[[pivot, row], :]
                order[row], order[pivot] = order[pivot], order[row]

            for r2 in range(m):
                if r2 != row and M[r2, col] == 1:
                    M[r2, :] ^= M[row, :]

            pivots += 1
            row += 1
            if row == m:
                break

        kept_rows = order[:pivots]
        A_ind = GF2.as_u8(A[kept_rows, :])
        return A_ind, kept_rows, pivots


# ============================================================
#  Pauli <-> binary (X|Z)
# ============================================================

class PauliBinary:
    PAULI_TO_XZ = {
        "I": (0, 0),
        "X": (1, 0),
        "Z": (0, 1),
        "Y": (1, 1),
    }

    @staticmethod
    def pauli_string_to_row(p: str) -> np.ndarray:
        n = len(p)
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)
        for i, ch in enumerate(p):
            if ch not in PauliBinary.PAULI_TO_XZ:
                raise ValueError(f"Invalid Pauli '{ch}' in '{p}'. Allowed: I,X,Y,Z.")
            xi, zi = PauliBinary.PAULI_TO_XZ[ch]
            x[i] = xi
            z[i] = zi
        return np.concatenate([x, z]).astype(np.uint8)

    @staticmethod
    def stabilizers_to_Hq(stabilizers: Sequence[str]) -> np.ndarray:
        if not stabilizers:
            raise ValueError("No stabilizers provided.")
        n = len(stabilizers[0])
        for s in stabilizers:
            if len(s) != n:
                raise ValueError("All stabilizer strings must have equal length.")
        H = np.vstack([PauliBinary.pauli_string_to_row(s) for s in stabilizers])
        return GF2.as_u8(H)


# ============================================================
#  Gaussian elimination over GF(2) on X-half (row ops only)
# ============================================================

@dataclass
class ElimResult:
    H: np.ndarray
    pivot_cols: List[int]
    r: int


class XHalfEliminator:
    """
    Row-reduce H over GF(2) using only row operations,
    pivoting on X-half columns [0..n-1]. Tracks pivot cols and r.
    """

    @staticmethod
    def row_reduce_X_half(Hq: np.ndarray, n: int) -> ElimResult:
        H = GF2.as_u8(Hq.copy())
        m = H.shape[0]
        pivots: List[int] = []
        row = 0

        for col in range(n):
            pivot = None
            for p in range(row, m):
                if H[p, col] == 1:
                    pivot = p
                    break
            if pivot is None:
                continue

            if pivot != row:
                H[[row, pivot], :] = H[[pivot, row], :]

            for r2 in range(m):
                if r2 != row and H[r2, col] == 1:
                    H[r2, :] ^= H[row, :]

            pivots.append(col)
            row += 1
            if row == m:
                break

        return ElimResult(H=H, pivot_cols=pivots, r=len(pivots))


# ============================================================
#  Standard-form builder (Hs-like) by choosing a qubit permutation
#  (Option B: enforce Eq.(18) I2 block in Z-bottom-mid)
# ============================================================

@dataclass
class StandardForm:
    Hs: np.ndarray
    n: int
    k: int
    r: int
    mid: int  # n-k-r

    perm_q: List[int]       # new qubit index -> old qubit index
    inv_perm_q: List[int]   # old -> new

    # extracted blocks (from chosen partition)
    A2: np.ndarray  # r x k
    C1: np.ndarray  # r x mid
    C2: np.ndarray  # r x k
    E: np.ndarray   # mid x k


class StandardFormBuilder:
    @staticmethod
    def choose_qubit_permutation(n: int, pivot_cols: List[int], k: int,
                                 logical_qubits: Optional[List[int]] = None) -> List[int]:
        piv = list(pivot_cols)
        r = len(piv)
        remaining = [j for j in range(n) if j not in piv]

        if len(remaining) < k:
            raise ValueError(
                f"Not enough non-pivot qubits to choose k={k} logical columns "
                f"(remaining={len(remaining)}, r={r}, n={n})."
            )

        if logical_qubits is not None:
            logical = list(logical_qubits)
            if len(logical) != k:
                raise ValueError("logical_qubits must have exactly k entries.")
            if any(q in piv for q in logical):
                raise ValueError("logical_qubits must not include pivot qubits.")
            if any(q < 0 or q >= n for q in logical):
                raise ValueError("logical_qubits out of range.")
        else:
            logical = sorted(remaining)[-k:]

        middle = [j for j in remaining if j not in logical]
        perm_q = piv + middle + logical
        if len(perm_q) != n:
            raise RuntimeError("Permutation construction failed.")
        return perm_q

    @staticmethod
    def apply_qubit_permutation(H: np.ndarray, perm_q: List[int]) -> np.ndarray:
        n = len(perm_q)
        X = H[:, :n]
        Z = H[:, n:]
        Xp = X[:, perm_q]
        Zp = Z[:, perm_q]
        return GF2.as_u8(np.hstack([Xp, Zp]))

    @staticmethod
    def extract_blocks(Hs: np.ndarray, n: int, k: int, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mid = n - k - r
        X = Hs[:, :n]
        Z = Hs[:, n:]

        top = slice(0, r)
        bot = slice(r, Hs.shape[0])

        g2 = slice(r, r + mid)
        g3 = slice(r + mid, n)

        A2 = GF2.as_u8(X[top, g3])  # r x k
        C1 = GF2.as_u8(Z[top, g2])  # r x mid
        C2 = GF2.as_u8(Z[top, g3])  # r x k
        E_raw = GF2.as_u8(Z[bot, g3])  # (m-r) x k

        if mid == 0:
            E = np.zeros((0, k), dtype=np.uint8)
        else:
            E = np.zeros((mid, k), dtype=np.uint8)
            take = min(mid, E_raw.shape[0])
            if take > 0:
                E[:take, :] = E_raw[:take, :]

        return A2, C1, C2, E

    @staticmethod
    def _rank_gf2(A: np.ndarray) -> int:
        """Small GF(2) rank helper (row-reduction, does not modify input)."""
        M = GF2.as_u8(A.copy())
        m, ncols = M.shape
        r = 0
        c = 0
        while r < m and c < ncols:
            piv = None
            for i in range(r, m):
                if M[i, c] == 1:
                    piv = i
                    break
            if piv is None:
                c += 1
                continue
            if piv != r:
                M[[r, piv], :] = M[[piv, r], :]
            for i in range(m):
                if i != r and M[i, c] == 1:
                    M[i, :] ^= M[r, :]
            r += 1
            c += 1
        return r

    @staticmethod
    def _force_bottom_Zmid_identity(H: np.ndarray, n: int, r: int, mid: int) -> tuple[Optional[np.ndarray], bool]:
        """
        Using ONLY row operations among the bottom rows (rows r..m-1),
        try to transform Z[bottom, g2] into an identity I_mid, where g2 = r..r+mid-1.
        Returns: (H_new or None, success)
        """
        if mid == 0:
            return GF2.as_u8(H), True

        H = GF2.as_u8(H.copy())
        m = H.shape[0]
        bot_rows = list(range(r, m))
        if len(bot_rows) < mid:
            return None, False

        Z = H[:, n:]
        g2_cols = list(range(r, r + mid))
        sub = GF2.as_u8(Z[bot_rows, :][:, g2_cols])

        if StandardFormBuilder._rank_gf2(sub) < mid:
            return None, False

        # Gaussian elimination on the submatrix, applying row ops to full H but restricted to bottom rows.
        row_pos = 0
        for col in g2_cols:
            piv = None
            for rr in range(row_pos, len(bot_rows)):
                if H[bot_rows[rr], n + col] == 1:
                    piv = rr
                    break
            if piv is None:
                continue

            if piv != row_pos:
                a = bot_rows[row_pos]
                b = bot_rows[piv]
                H[[a, b], :] = H[[b, a], :]
                bot_rows[row_pos], bot_rows[piv] = bot_rows[piv], bot_rows[row_pos]

            piv_global = bot_rows[row_pos]
            for rr in range(len(bot_rows)):
                if rr == row_pos:
                    continue
                rr_global = bot_rows[rr]
                if H[rr_global, n + col] == 1:
                    H[rr_global, :] ^= H[piv_global, :]

            row_pos += 1
            if row_pos == mid:
                break

        Z = H[:, n:]
        sub2 = GF2.as_u8(Z[bot_rows, :][:, g2_cols])

        ok = True
        for i in range(mid):
            for j in range(mid):
                if sub2[i, j] != (1 if i == j else 0):
                    ok = False
                    break
            if not ok:
                break

        return (H if ok else None), ok

    @staticmethod
    def _candidate_logical_sets(remaining: List[int], k: int) -> List[List[int]]:
        """Generate candidate logical-column choices."""
        if k == 0:
            return [[]]
        rem = list(remaining)

        # small-k brute force (Steane: k=1)
        if k <= 2 and len(rem) <= 32:
            return [list(c) for c in itertools.combinations(rem, k)]

        # heuristics
        cands = [sorted(rem)[-k:], sorted(rem)[:k]]
        rem_sorted = sorted(rem)
        for shift in range(min(8, len(rem_sorted))):
            cand = rem_sorted[shift:shift + k]
            if len(cand) == k:
                cands.append(cand)

        out: List[List[int]] = []
        seen = set()
        for c in cands:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                out.append(c)
        return out

    @staticmethod
    def build(Hq: np.ndarray, n: int, k: int, logical_qubits: Optional[List[int]] = None) -> StandardForm:
        """Build Hs in the paper's Eq.(18) standard form."""
        elim1 = XHalfEliminator.row_reduce_X_half(Hq, n)
        piv = list(elim1.pivot_cols)
        r0 = elim1.r

        remaining = [j for j in range(n) if j not in piv]
        if len(remaining) < k:
            raise ValueError(
                f"Not enough non-pivot qubits to choose k={k} logical columns "
                f"(remaining={len(remaining)}, r={r0}, n={n})."
            )

        if logical_qubits is not None:
            logical_sets = [list(logical_qubits)]
        else:
            logical_sets = StandardFormBuilder._candidate_logical_sets(remaining, k)

        best = None

        for logical in logical_sets:
            if len(logical) != k:
                continue
            if any(q in piv for q in logical):
                continue

            middle = [j for j in remaining if j not in logical]
            perm_q = piv + middle + logical
            if len(perm_q) != n:
                continue

            Hp = StandardFormBuilder.apply_qubit_permutation(elim1.H, perm_q)
            elim2 = XHalfEliminator.row_reduce_X_half(Hp, n)
            Hs_try = elim2.H
            r = elim2.r
            mid = n - k - r
            if mid < 0:
                continue

            Hs_fixed, ok = StandardFormBuilder._force_bottom_Zmid_identity(Hs_try, n, r, mid)
            if not ok:
                continue

            best = (Hs_fixed, perm_q, r, mid)
            break

        if best is None:
            raise RuntimeError(
                "Failed to build Hs in Eq.(18) standard form. Try specifying logical_qubits."
            )

        Hs, perm_q, r, mid = best

        inv_perm = [0] * n
        for new_i, old_i in enumerate(perm_q):
            inv_perm[old_i] = new_i

        A2, C1, C2, E = StandardFormBuilder.extract_blocks(Hs, n, k, r)

        return StandardForm(
            Hs=Hs, n=n, k=k, r=r, mid=mid,
            perm_q=perm_q, inv_perm_q=inv_perm,
            A2=A2, C1=C1, C2=C2, E=E
        )


# ============================================================
#  Logical operators from Eq.(19)-(20)
# ============================================================

@dataclass
class LogicalOps:
    Xbars: List[np.ndarray]
    Zbars: List[np.ndarray]
    Xbars_perm: List[np.ndarray]
    Zbars_perm: List[np.ndarray]


class LogicalOperatorBuilder:
    @staticmethod
    def compute(sf: StandardForm) -> LogicalOps:
        n, k, r, mid = sf.n, sf.k, sf.r, sf.mid
        A2, C1, C2, E = sf.A2, sf.C1, sf.C2, sf.E

        U2 = GF2.as_u8(E.T)                 # k x mid
        U3 = np.eye(k, dtype=np.uint8)      # k x k

        # V1 = E^T C1^T + C2^T  (transpose on C1 is important)
        C1T = GF2.as_u8(C1.T)               # mid x r
        term = GF2.as_u8(U2 @ C1T)          # k x r (int matmul then mod2)
        V1 = GF2.add(term, GF2.as_u8(C2.T)) # k x r

        V1p = GF2.as_u8(A2.T)               # k x r
        V3p = np.eye(k, dtype=np.uint8)     # k x k

        Xbars_perm: List[np.ndarray] = []
        Zbars_perm: List[np.ndarray] = []

        for i in range(k):
            x = np.concatenate([
                np.zeros(r, dtype=np.uint8),
                U2[i, :] if mid > 0 else np.zeros(0, dtype=np.uint8),
                U3[i, :],
            ])
            z = np.concatenate([
                V1[i, :],
                np.zeros(mid, dtype=np.uint8),
                np.zeros(k, dtype=np.uint8),
            ])
            Xbars_perm.append(GF2.as_u8(np.concatenate([x, z])))

            xz = np.zeros(n, dtype=np.uint8)
            zz = np.concatenate([
                V1p[i, :],
                np.zeros(mid, dtype=np.uint8),
                V3p[i, :],
            ])
            Zbars_perm.append(GF2.as_u8(np.concatenate([xz, zz])))

        def unpermute(v: np.ndarray) -> np.ndarray:
            x, z = v[:n], v[n:]
            x_un = np.zeros(n, dtype=np.uint8)
            z_un = np.zeros(n, dtype=np.uint8)
            for new_i, old_i in enumerate(sf.perm_q):
                x_un[old_i] = x[new_i]
                z_un[old_i] = z[new_i]
            return GF2.as_u8(np.concatenate([x_un, z_un]))

        Xbars = [unpermute(v) for v in Xbars_perm]
        Zbars = [unpermute(v) for v in Zbars_perm]

        return LogicalOps(
            Xbars=Xbars,
            Zbars=Zbars,
            Xbars_perm=Xbars_perm,
            Zbars_perm=Zbars_perm
        )


# ============================================================
#  Checks
# ============================================================

class StabilizerChecks:
    @staticmethod
    def commutes_with_all(Hq: np.ndarray, op: np.ndarray, n: int) -> bool:
        Hq = GF2.as_u8(Hq)
        op = GF2.as_u8(op)
        for row in Hq:
            if GF2.symplectic_inner(row, op, n) != 0:
                return False
        return True

    @staticmethod
    def anticommutes(u: np.ndarray, v: np.ndarray, n: int) -> bool:
        return GF2.symplectic_inner(u, v, n) == 1


# ============================================================
#  Pretty printing helpers
# ============================================================

class Pretty:
    @staticmethod
    def bits_to_str(bits: np.ndarray) -> str:
        return "".join(str(int(b)) for b in bits.tolist())

    @staticmethod
    def print_stabilizers(stabilizers: Sequence[str]) -> None:
        print("Input stabilizers:")
        for s in stabilizers:
            print(" ", s)

    @staticmethod
    def print_matrix_XZ(H: np.ndarray, title: str = "H") -> None:
        H = GF2.as_u8(H)
        n = H.shape[1] // 2
        print(f"{title} (X|Z):")
        for row in H:
            x = Pretty.bits_to_str(row[:n])
            z = Pretty.bits_to_str(row[n:])
            print(f"  {x} | {z}")

    @staticmethod
    def print_logicals(logops: LogicalOps, n: int) -> None:
        for i in range(len(logops.Xbars)):
            xb = logops.Xbars[i]
            zb = logops.Zbars[i]
            print(f"logical qubit {i}:")
            print(f"  Xbar[{i}] = {Pretty.bits_to_str(xb[:n])} | {Pretty.bits_to_str(xb[n:])}")
            print(f"  Zbar[{i}] = {Pretty.bits_to_str(zb[:n])} | {Pretty.bits_to_str(zb[n:])}")


# ============================================================
#  Pipeline
# ============================================================

class StabilizerPipeline:
    @staticmethod
    def run(
        stabilizers: Optional[Sequence[str]] = None,
        Hq: Optional[np.ndarray] = None,
        *,
        n: Optional[int] = None,
        k: Optional[int] = None,
        logical_qubits: Optional[List[int]] = None,
        reduce_dependent: bool = True,
    ) -> Tuple[np.ndarray, StandardForm, LogicalOps, int]:
        """
        Returns: (Hq_used, sf, logops, k_used)
        """
        if (stabilizers is None) == (Hq is None):
            raise ValueError("Provide exactly one of: stabilizers OR Hq.")

        # ---------- build / accept Hq ----------
        if stabilizers is not None:
            Pretty.print_stabilizers(stabilizers)
            Hq_in = PauliBinary.stabilizers_to_Hq(stabilizers)
            n_used = len(stabilizers[0])
        else:
            Hq_in = GF2.as_u8(Hq)
            if Hq_in.ndim != 2:
                raise ValueError("Hq must be a 2D array.")
            if Hq_in.shape[1] % 2 != 0:
                raise ValueError("Hq must have 2n columns.")
            n_used = Hq_in.shape[1] // 2
            if n is not None and n != n_used:
                raise ValueError(f"Provided n={n} but inferred n={n_used} from Hq shape.")

        print()

        # ---------- reduce dependent rows (optional but recommended) ----------
        if reduce_dependent:
            Hq_used, kept_rows, rank_val = GF2Rank.independent_rows(Hq_in)
            if Hq_used.shape[0] != Hq_in.shape[0]:
                print(f"Reduced dependent stabilizers: kept {Hq_used.shape[0]}/{Hq_in.shape[0]} rows")
                print(f"Kept original row indices: {kept_rows}")
            else:
                print("No dependent stabilizers detected (all rows independent).")
        else:
            Hq_used = Hq_in
            rank_val = GF2Rank.rank(Hq_used)
            kept_rows = list(range(Hq_used.shape[0]))

        # ---------- infer k if not provided ----------
        if k is None:
            k_used = n_used - rank_val
            if k_used < 0:
                raise ValueError(f"Inferred k is negative: n={n_used}, rank={rank_val}. Check Hq.")
            print(f"Inferred rank(Hq) = {rank_val}")
            print(f"Inferred k = n - rank = {n_used} - {rank_val} = {k_used}")
        else:
            k_used = int(k)
            print(f"Using provided k = {k_used}")
            print(f"Computed rank(Hq) = {rank_val} (for reference)")

        print()

        # ---------- print Hq ----------
        Pretty.print_matrix_XZ(Hq_used, title="Hq (independent rows)" if reduce_dependent else "Hq")
        print()

        # ---------- build Hs-like form ----------
        sf = StandardFormBuilder.build(Hq_used, n=n_used, k=k_used, logical_qubits=logical_qubits)

        print(f"Computed r = {sf.r}")
        print(f"Computed mid = n-k-r = {sf.mid}")
        print(f"Qubit permutation (new -> old) used to form Hs: {sf.perm_q}")
        print()

        Pretty.print_matrix_XZ(sf.Hs, title="Hs (permuted qubit order)")
        print()

        # ---------- compute logicals ----------
        logops = LogicalOperatorBuilder.compute(sf)
        Pretty.print_logicals(logops, n_used)
        print()

        # ---------- checks ----------
        for i in range(k_used):
            okX = StabilizerChecks.commutes_with_all(Hq_used, logops.Xbars[i], n_used)
            okZ = StabilizerChecks.commutes_with_all(Hq_used, logops.Zbars[i], n_used)
            okXZ = StabilizerChecks.anticommutes(logops.Xbars[i], logops.Zbars[i], n_used)

            print(f"Checks for logical qubit {i}:")
            print(f"  Xbar commutes with all stabilizers: {okX}")
            print(f"  Zbar commutes with all stabilizers: {okZ}")
            print(f"  Xbar anticommutes with Zbar:       {okXZ}")
            print()

        return Hq_used, sf, logops, k_used