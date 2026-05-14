# stabilizer_logicals_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple
import itertools
import numpy as np


# ============================================================
# GF(2) utilities + symplectic product
# ============================================================

class GF2:
    @staticmethod
    def as_u8(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=np.uint8) & 1)

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
    @staticmethod
    def independent_rows(A: np.ndarray) -> Tuple[np.ndarray, List[int], int]:
        """
        Return (A_ind, kept_row_indices, rank) via GF(2) elimination.
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

        kept = row_map[:rank]
        return GF2.as_u8(A[kept, :]), kept, rank

    @staticmethod
    def rank(A: np.ndarray) -> int:
        _, _, r = GF2Rank.independent_rows(A)
        return r


# ============================================================
# Pauli <-> binary (X|Z)
# ============================================================

class PauliBinary:
    PAULI_TO_XZ = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}

    @staticmethod
    def pauli_string_to_row(p: str) -> np.ndarray:
        n = len(p)
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)
        for i, ch in enumerate(p):
            if ch not in PauliBinary.PAULI_TO_XZ:
                raise ValueError(f"Invalid Pauli '{ch}'. Allowed: I,X,Y,Z.")
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
                raise ValueError("All stabilizers must have equal length.")
        H = np.vstack([PauliBinary.pauli_string_to_row(s) for s in stabilizers])
        return GF2.as_u8(H)


# ============================================================
# Canonical column sorting (signature = (Xcol bits, Zcol bits))
# ============================================================

class Canonicalizer:
    @staticmethod
    def canonical_column_order(Hq: np.ndarray, n: int) -> List[int]:
        """
        Deterministic column order:
          signature(j) = (X[:,j] bits..., Z[:,j] bits...)
        Lexicographic ascending, stable ties by column index.
        """
        Hq = GF2.as_u8(Hq)
        X = Hq[:, :n]
        Z = Hq[:, n:]

        sigs = []
        for j in range(n):
            sig = tuple(int(b) for b in (X[:, j].tolist() + Z[:, j].tolist()))
            sigs.append((sig, j))

        sigs.sort(key=lambda t: (t[0], t[1]))
        return [j for (_, j) in sigs]

    @staticmethod
    def apply_qubit_permutation(Hq: np.ndarray, perm_q: List[int]) -> np.ndarray:
        """
        perm_q: new -> old column indices (qubits).
        Applies same permutation to X and Z halves.
        """
        Hq = GF2.as_u8(Hq)
        n = len(perm_q)
        X = Hq[:, :n]
        Z = Hq[:, n:]
        Xp = X[:, perm_q]
        Zp = Z[:, perm_q]
        return GF2.as_u8(np.hstack([Xp, Zp]))


# ============================================================
# Deterministic elimination on X-half
# (leftmost pivot column, topmost pivot row)
# ============================================================

@dataclass
class ElimResult:
    H: np.ndarray
    pivot_cols: List[int]
    r: int


class XHalfEliminator:
    @staticmethod
    def row_reduce_X_half(Hq: np.ndarray, n: int) -> ElimResult:
        H = GF2.as_u8(Hq.copy())
        m = H.shape[0]
        pivots: List[int] = []
        row = 0

        for col in range(n):  # leftmost pivot col
            pivot = None
            for p in range(row, m):  # topmost pivot row
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
# StandardForm + enforcing Eq.(18) bottom Z-mid = I
# ============================================================

@dataclass
class StandardForm:
    Hs: np.ndarray
    n: int
    k: int
    r: int
    mid: int
    perm_q: List[int]        # new -> old (original qubit indices)
    inv_perm_q: List[int]    # old -> new

    # blocks for Eq.(19)-(20)
    A2: np.ndarray  # r x k
    C1: np.ndarray  # r x mid
    C2: np.ndarray  # r x k
    E: np.ndarray   # mid x k


class StandardFormBuilder:
    @staticmethod
    def _rank_gf2(A: np.ndarray) -> int:
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
    def _force_bottom_Zmid_identity(H: np.ndarray, n: int, r: int, mid: int) -> Tuple[Optional[np.ndarray], bool]:
        """
        Force Z[ r:r+mid , g2 ] = I_mid using ONLY row operations on rows r..m-1.
        Here g2 are the mid qubit columns in the current [piv | mid | logical] ordering,
        i.e. g2 = [r, ..., r+mid-1] (in qubit indices).

        Robust Gauss–Jordan on the square (mid x mid) submatrix using rows r..r+mid-1.
        """
        if mid == 0:
            return GF2.as_u8(H), True

        H = GF2.as_u8(H.copy())
        m = H.shape[0]
        if m < r + mid:
            return None, False  # not enough bottom rows

        g2 = list(range(r, r + mid))
        row_base = r

        Z = H[:, n:]
        S = GF2.as_u8(Z[row_base:row_base + mid, :][:, g2])

        if StandardFormBuilder._rank_gf2(S) < mid:
            return None, False

        for col in range(mid):
            piv = None
            for rr in range(col, mid):
                if S[rr, col] == 1:
                    piv = rr
                    break
            if piv is None:
                return None, False

            if piv != col:
                S[[col, piv], :] = S[[piv, col], :]
                H[[row_base + col, row_base + piv], :] = H[[row_base + piv, row_base + col], :]

            for rr in range(mid):
                if rr == col:
                    continue
                if S[rr, col] == 1:
                    S[rr, :] ^= S[col, :]
                    H[row_base + rr, :] ^= H[row_base + col, :]

        Z2 = H[:, n:]
        S2 = GF2.as_u8(Z2[row_base:row_base + mid, :][:, g2])
        if not np.array_equal(S2, np.eye(mid, dtype=np.uint8)):
            return None, False

        return H, True

    @staticmethod
    def build_canonical(Hq: np.ndarray, n: int, k: int) -> StandardForm:
        """
        Canonical pipeline (deterministic, no brute force permutations):
          1) canonical column sort by (Xcol,Zcol) signature
          2) deterministic elimination on X-half
          3) deterministically choose (mid,logical) partition of nonpivot columns
             such that rank(Z_bottom,mid)=mid
          4) permute to [pivots | mid | logical]
          5) enforce bottom Z-mid = I (Eq.18) using bottom-row ops only
          6) extract blocks
        """
        Hq = GF2.as_u8(Hq)

        # (1) canonical signature sort
        perm_sig = Canonicalizer.canonical_column_order(Hq, n)
        H1 = Canonicalizer.apply_qubit_permutation(Hq, perm_sig)

        # (2) deterministic X-half elimination
        elim1 = XHalfEliminator.row_reduce_X_half(H1, n)
        H_elim = elim1.H
        piv = list(elim1.pivot_cols)
        r = elim1.r

        mid = n - k - r
        if mid < 0:
            raise RuntimeError("Computed mid < 0; inconsistent parameters.")

        nonpiv = [j for j in range(n) if j not in piv]
        if len(nonpiv) < k:
            raise ValueError(f"Cannot choose k={k} logical columns from {len(nonpiv)} nonpivot columns.")

        # Rank helper on current eliminated matrix coords
        def rank_Zbot_of_g2(g2_cols: List[int]) -> int:
            if mid == 0:
                return 0
            m = H_elim.shape[0]
            bot = list(range(r, m))
            Z = H_elim[:, n:]
            sub = GF2.as_u8(Z[bot, :][:, g2_cols])
            return StandardFormBuilder._rank_gf2(sub)

        # (3) rank-aware deterministic partition selection
        chosen_logical: Optional[List[int]] = None
        chosen_middle: Optional[List[int]] = None

        if k == 0:
            chosen_logical = []
            chosen_middle = nonpiv[:]
        elif k == 1:
            # try logical column candidates from rightmost->leftmost (deterministic)
            for logcol in reversed(nonpiv):
                middle = [c for c in nonpiv if c != logcol]
                if len(middle) != mid:
                    continue
                if rank_Zbot_of_g2(middle) == mid:
                    chosen_logical = [logcol]
                    chosen_middle = middle
                    break
        else:
            # deterministic combinations in lex order
            for logical_cols in itertools.combinations(nonpiv, k):
                logical_cols = list(logical_cols)
                middle = [c for c in nonpiv if c not in logical_cols]
                if len(middle) != mid:
                    continue
                if rank_Zbot_of_g2(middle) == mid:
                    chosen_logical = logical_cols
                    chosen_middle = middle
                    break

        if chosen_logical is None or chosen_middle is None:
            raise RuntimeError(
                "No valid (mid,logical) partition found with rank(Z_bottom,mid)=mid under canonical column sort."
            )

        perm_partition = piv + chosen_middle + chosen_logical

        # (4) permute into [piv|mid|logical]
        H2 = Canonicalizer.apply_qubit_permutation(H_elim, perm_partition)

        # (5) enforce Eq.(18)
        Hs, ok = StandardFormBuilder._force_bottom_Zmid_identity(H2, n, r, mid)
        if not ok or Hs is None:
            raise RuntimeError("Failed to enforce bottom Z-mid identity (Eq.18).")

        # Compose overall perm relative to ORIGINAL qubits
        overall_perm = [perm_sig[j] for j in perm_partition]

        inv_perm = [0] * n
        for new_i, old_i in enumerate(overall_perm):
            inv_perm[old_i] = new_i

        A2, C1, C2, E = StandardFormBuilder.extract_blocks(Hs, n, k, r)

        return StandardForm(
            Hs=Hs, n=n, k=k, r=r, mid=mid,
            perm_q=overall_perm, inv_perm_q=inv_perm,
            A2=A2, C1=C1, C2=C2, E=E
        )


# ============================================================
# Logical operators from Eq.(19)-(20)
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

        # U2 = E^T, U3 = I_k
        U2 = GF2.as_u8(E.T)              # k x mid
        U3 = np.eye(k, dtype=np.uint8)   # k x k

        # V1 = E^T C1^T + C2^T
        V1 = (U2 @ GF2.as_u8(C1.T)) & 1
        V1 = (V1 ^ GF2.as_u8(C2.T)) & 1  # k x r

        # V1' = A2^T, V3' = I_k
        V1p = GF2.as_u8(A2.T)            # k x r
        V3p = np.eye(k, dtype=np.uint8)  # k x k

        Xbars_perm: List[np.ndarray] = []
        Zbars_perm: List[np.ndarray] = []

        for i in range(k):
            # Xbar (perm coords): [0_r, U2_row, U3_row | V1_row, 0_mid, 0_k]
            x = np.concatenate([
                np.zeros(r, dtype=np.uint8),
                U2[i, :] if mid > 0 else np.zeros((0,), dtype=np.uint8),
                U3[i, :]
            ])
            z = np.concatenate([
                GF2.as_u8(V1[i, :]),
                np.zeros(mid, dtype=np.uint8),
                np.zeros(k, dtype=np.uint8)
            ])
            Xbars_perm.append(GF2.as_u8(np.concatenate([x, z])))

            # Zbar (perm coords): [0_n | V1'_row, 0_mid, V3'_row]
            xz = np.zeros(n, dtype=np.uint8)
            zz = np.concatenate([
                GF2.as_u8(V1p[i, :]),
                np.zeros(mid, dtype=np.uint8),
                V3p[i, :]
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

        return LogicalOps(Xbars=Xbars, Zbars=Zbars, Xbars_perm=Xbars_perm, Zbars_perm=Zbars_perm)


# ============================================================
# Checks + pretty printing
# ============================================================

class StabilizerChecks:
    @staticmethod
    def commutes_with_all(Hq: np.ndarray, logical: np.ndarray, n: int) -> bool:
        Hq = GF2.as_u8(Hq)
        logical = GF2.as_u8(logical)
        for row in Hq:
            if GF2.symplectic_product(row, logical, n) != 0:
                return False
        return True

    @staticmethod
    def anticommutes(Xbar: np.ndarray, Zbar: np.ndarray, n: int) -> bool:
        return GF2.symplectic_product(Xbar, Zbar, n) == 1


class Pretty:
    @staticmethod
    def bits(v: np.ndarray) -> str:
        v = GF2.as_u8(v).ravel()
        return "".join(str(int(b)) for b in v.tolist())

    @staticmethod
    def row_XZ(row: np.ndarray, n: int) -> str:
        row = GF2.as_u8(row).ravel()
        return f"{Pretty.bits(row[:n])} | {Pretty.bits(row[n:])}"

    @staticmethod
    def print_matrix_XZ(H: np.ndarray, title: str):
        H = GF2.as_u8(H)
        n = H.shape[1] // 2
        print(f"{title} (X|Z):")
        for r in H:
            print(" ", Pretty.row_XZ(r, n))

    @staticmethod
    def print_logicals(logops: LogicalOps, n: int):
        for i, (x, z) in enumerate(zip(logops.Xbars, logops.Zbars)):
            print(f"logical qubit {i}:")
            print(f"  Xbar[{i}] = {Pretty.row_XZ(x, n)}")
            print(f"  Zbar[{i}] = {Pretty.row_XZ(z, n)}")


# ============================================================
# Enumerate "canonical family" base logical pairs (no brute perms)
# ============================================================

@dataclass(frozen=True)
class CandidateLogicalPair:
    perm_q: tuple[int, ...]             # overall perm (new -> old) in ORIGINAL coordinates
    logical_cols_sig: tuple[int, ...]   # which logical cols chosen among nonpiv (in signature-sorted coords)
    r: int
    mid: int
    Xbar: tuple[int, ...]              # length 2n (original coords)
    Zbar: tuple[int, ...]              # length 2n (original coords)


def _tuple_bits(v: np.ndarray) -> tuple[int, ...]:
    v = GF2.as_u8(v).ravel()
    return tuple(int(b) for b in v.tolist())


def format_xz(bits_2n: tuple[int, ...], n: int) -> str:
    x = "".join(str(b) for b in bits_2n[:n])
    z = "".join(str(b) for b in bits_2n[n:])
    return f"{x} | {z}"


def enumerate_logical_pairs_canonical_family(Hq: np.ndarray, n: int, k: int) -> List[CandidateLogicalPair]:
    """
    Fix:
      - canonical signature column order
      - deterministic X-half elimination => fixed pivots

    Enumerate:
      - all choices of which k nonpivot columns become g3 (logical)
        rest become g2 (mid)

    Keep only those partitions that can be brought to Eq.(18).
    Deduplicate by (Xbar,Zbar) in original coordinates.
    """
    Hq = GF2.as_u8(Hq)

    perm_sig = Canonicalizer.canonical_column_order(Hq, n)
    H1 = Canonicalizer.apply_qubit_permutation(Hq, perm_sig)

    elim1 = XHalfEliminator.row_reduce_X_half(H1, n)
    H_elim = elim1.H
    piv = list(elim1.pivot_cols)
    r = elim1.r
    mid = n - k - r
    if mid < 0:
        return []

    nonpiv = [j for j in range(n) if j not in piv]
    if len(nonpiv) < k:
        return []

    logical_choices = [()] if k == 0 else list(itertools.combinations(nonpiv, k))

    seen = set()
    out: List[CandidateLogicalPair] = []

    for logical_cols in logical_choices:
        logical_cols = tuple(logical_cols)
        middle_cols = [c for c in nonpiv if c not in logical_cols]
        if len(middle_cols) != mid:
            continue

        perm_partition = piv + middle_cols + list(logical_cols)
        H2 = Canonicalizer.apply_qubit_permutation(H_elim, perm_partition)

        Hs, ok = StandardFormBuilder._force_bottom_Zmid_identity(H2, n, r, mid)
        if not ok or Hs is None:
            continue

        overall_perm = [perm_sig[j] for j in perm_partition]
        inv_perm = [0] * n
        for new_i, old_i in enumerate(overall_perm):
            inv_perm[old_i] = new_i

        A2, C1, C2, E = StandardFormBuilder.extract_blocks(Hs, n, k, r)
        sf_tmp = StandardForm(
            Hs=Hs, n=n, k=k, r=r, mid=mid,
            perm_q=overall_perm, inv_perm_q=inv_perm,
            A2=A2, C1=C1, C2=C2, E=E
        )

        logops = LogicalOperatorBuilder.compute(sf_tmp)

        if k == 0:
            X_t = tuple([0] * (2 * n))
            Z_t = tuple([0] * (2 * n))
        else:
            Xb = logops.Xbars[0]
            Zb = logops.Zbars[0]

            okX = StabilizerChecks.commutes_with_all(Hq, Xb, n)
            okZ = StabilizerChecks.commutes_with_all(Hq, Zb, n)
            okXZ = StabilizerChecks.anticommutes(Xb, Zb, n)
            if not (okX and okZ and okXZ):
                continue

            X_t = _tuple_bits(Xb)
            Z_t = _tuple_bits(Zb)

        key = (X_t, Z_t)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            CandidateLogicalPair(
                perm_q=tuple(overall_perm),
                logical_cols_sig=logical_cols,
                r=r,
                mid=mid,
                Xbar=X_t,
                Zbar=Z_t,
            )
        )

    return out


# ============================================================
# Expand by stabilizer cosets: (X,Z) -> (X+s, Z+t)
# ============================================================

def gf2_rowspace_basis_rows(A: np.ndarray) -> List[np.ndarray]:
    """
    Return a basis for the row space of A over GF(2), as explicit row vectors.
    Deterministic (leftmost pivot column, topmost pivot row).
    """
    A = GF2.as_u8(A.copy())
    m, n = A.shape
    row = 0
    basis_rows: List[np.ndarray] = []

    for col in range(n):
        piv = None
        for rr in range(row, m):
            if A[rr, col] == 1:
                piv = rr
                break
        if piv is None:
            continue

        if piv != row:
            A[[row, piv], :] = A[[piv, row], :]

        for rr in range(m):
            if rr != row and A[rr, col] == 1:
                A[rr, :] ^= A[row, :]

        basis_rows.append(A[row, :].copy())
        row += 1
        if row == m:
            break

    return basis_rows


def enumerate_stabilizer_equivalent_pairs(
    Hq: np.ndarray,
    n: int,
    Xbar: np.ndarray,
    Zbar: np.ndarray,
    *,
    max_basis: int = 18,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Enumerate (Xbar + s, Zbar + t) for all s,t in span(stabilizers).
    This makes X vary (and can recover paper reps) without perm brute force.

    Safety: if basis dimension > max_basis, we cap enumeration to avoid explosion.
    """
    Hq = GF2.as_u8(Hq)
    Xbar = GF2.as_u8(Xbar).ravel()
    Zbar = GF2.as_u8(Zbar).ravel()

    basis = gf2_rowspace_basis_rows(Hq)
    d = len(basis)
    if d > max_basis:
        # deterministic cap: take first max_basis basis vectors
        basis = basis[:max_basis]
        d = max_basis

    # precompute all combinations of basis -> stabilizer row combo
    combos: List[np.ndarray] = []
    for mask in range(1 << d):
        v = np.zeros(2 * n, dtype=np.uint8)
        for i in range(d):
            if (mask >> i) & 1:
                v ^= basis[i]
        combos.append(v)

    out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    seen = set()

    for s in combos:
        Xp = Xbar ^ s
        # Xp still commutes with stabilizers and anticomm with Zbar because Zbar commutes with stabilizers.
        Xt = tuple(int(b) for b in Xp.tolist())
        for t in combos:
            Zp = Zbar ^ t
            Zt = tuple(int(b) for b in Zp.tolist())
            key = (Xt, Zt)
            if key in seen:
                continue
            # Keep only pairs that remain a logical pair (commute with stabilizers, anticommute with each other).
            # Commutes: Xp, Zp commute with Hq automatically if Hq rows mutually commute (valid stabilizer code).
            # Still, we check to be safe with possibly non-commuting inputs.
            okX = StabilizerChecks.commutes_with_all(Hq, np.array(Xp, dtype=np.uint8), n)
            okZ = StabilizerChecks.commutes_with_all(Hq, np.array(Zp, dtype=np.uint8), n)
            okXZ = StabilizerChecks.anticommutes(np.array(Xp, dtype=np.uint8), np.array(Zp, dtype=np.uint8), n)
            if not (okX and okZ and okXZ):
                continue

            seen.add(key)
            out.append(key)

    return out


# ============================================================
# Pipeline
# ============================================================

class StabilizerPipeline:
    @staticmethod
    def run(
        *,
        stabilizers: Optional[Sequence[str]] = None,
        Hq: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        k: Optional[int] = None,
        reduce_dependent: bool = True,
        list_all_logical_pairs: bool = True,
        expand_by_stabilizers: bool = True,
    ):
        if (stabilizers is None) == (Hq is None):
            raise ValueError("Provide exactly one of: stabilizers OR Hq.")

        # build/accept Hq
        if stabilizers is not None:
            Hq_in = PauliBinary.stabilizers_to_Hq(stabilizers)
            n_used = len(stabilizers[0])
        else:
            Hq_in = GF2.as_u8(Hq)
            if Hq_in.ndim != 2 or (Hq_in.shape[1] % 2) != 0:
                raise ValueError("Hq must be a 2D array with 2n columns.")
            n_used = Hq_in.shape[1] // 2
            if n is not None and n != n_used:
                raise ValueError(f"Provided n={n} but inferred n={n_used} from Hq shape.")

        print()

        # reduce dependent stabilizers
        if reduce_dependent:
            Hq_used, _, rank_val = GF2Rank.independent_rows(Hq_in)
            if Hq_used.shape[0] == Hq_in.shape[0]:
                print("No dependent stabilizers detected (all rows independent).")
            else:
                print(f"Reduced dependent stabilizers: kept {Hq_used.shape[0]}/{Hq_in.shape[0]} rows")
        else:
            Hq_used = Hq_in
            rank_val = GF2Rank.rank(Hq_used)

        # infer k
        if k is None:
            k_used = n_used - rank_val
            print(f"Inferred rank(Hq) = {rank_val}")
            print(f"Inferred k = n - rank = {n_used} - {rank_val} = {k_used}")
        else:
            k_used = int(k)
            print(f"Using provided k = {k_used}")
            print(f"Computed rank(Hq) = {rank_val} (for reference)")

        print()
        Pretty.print_matrix_XZ(Hq_used, "Hq (independent rows)" if reduce_dependent else "Hq")
        print()

        # canonical build
        sf = StandardFormBuilder.build_canonical(Hq_used, n=n_used, k=k_used)

        print(f"Computed r = {sf.r}")
        print(f"Computed mid = n-k-r = {sf.mid}")
        print(f"Canonical qubit permutation (new -> old): {sf.perm_q}")
        print()

        Pretty.print_matrix_XZ(sf.Hs, "Hs (canonical)")
        print()

        # logicals
        logops = LogicalOperatorBuilder.compute(sf)
        Pretty.print_logicals(logops, n_used)
        print()

        # checks
        for i in range(k_used):
            okX = StabilizerChecks.commutes_with_all(Hq_used, logops.Xbars[i], n_used)
            okZ = StabilizerChecks.commutes_with_all(Hq_used, logops.Zbars[i], n_used)
            okXZ = StabilizerChecks.anticommutes(logops.Xbars[i], logops.Zbars[i], n_used)
            print(f"Checks for logical qubit {i}:")
            print(f"  Xbar commutes with all stabilizers: {okX}")
            print(f"  Zbar commutes with all stabilizers: {okZ}")
            print(f"  Xbar anticommutes with Zbar:       {okXZ}")
            print()

        # Paper target (what you said the paper reports)
        # X = 0001101 | 0000000
        # Z = 0000000 | 0110001
        paper_X = tuple(int(b) for b in "00011010000000")  # 14 bits total (2n=14)
        paper_Z = tuple(int(b) for b in "00000000110001")  # 14 bits total

        # list all pairs
        if list_all_logical_pairs and k_used == 1:
            base_candidates = enumerate_logical_pairs_canonical_family(Hq_used, n_used, k_used)
            chosen_key = (_tuple_bits(logops.Xbars[0]), _tuple_bits(logops.Zbars[0]))

            print(f"Found {len(base_candidates)} base logical (Xbar,Zbar) pairs in the canonical family.")
            print("Now expanding each base pair by multiplying by stabilizers (i.e., adding stabilizer row combinations)...\n")

            all_pairs_set = set()
            all_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

            # expand
            for bc in base_candidates:
                Xb = np.array(bc.Xbar, dtype=np.uint8)
                Zb = np.array(bc.Zbar, dtype=np.uint8)

                if expand_by_stabilizers:
                    expanded = enumerate_stabilizer_equivalent_pairs(Hq_used, n_used, Xb, Zb, max_basis=18)
                else:
                    expanded = [(bc.Xbar, bc.Zbar)]

                for pair in expanded:
                    if pair in all_pairs_set:
                        continue
                    all_pairs_set.add(pair)
                    all_pairs.append(pair)

            # deterministic sort: by (wt, X, Z)
            def weight(bits2n: Tuple[int, ...]) -> int:
                return sum(bits2n)

            all_pairs_sorted = sorted(all_pairs, key=lambda p: (weight(p[0]) + weight(p[1]), p[0], p[1]))

            print(f"Total unique logical pairs after stabilizer expansion: {len(all_pairs_sorted)}")
            print("Listing all pairs. Marks: '*' = chosen by build_canonical().  'PAPER' = matches paper example.\n")

            for idx, (X_t, Z_t) in enumerate(all_pairs_sorted):
                mark = "*" if (X_t, Z_t) == chosen_key else " "
                mark2 = " PAPER" if (X_t == paper_X and Z_t == paper_Z) else ""
                print(f"{mark} [{idx:03d}]{mark2}")
                print(f"     Xbar = {format_xz(X_t, n_used)}")
                print(f"     Zbar = {format_xz(Z_t, n_used)}")
                print()

        return Hq_used, sf, logops, k_used