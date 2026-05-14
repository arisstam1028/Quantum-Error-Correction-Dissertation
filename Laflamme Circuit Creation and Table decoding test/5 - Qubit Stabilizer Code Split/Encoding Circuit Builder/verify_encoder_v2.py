# verify_encoder_v2.py
from __future__ import annotations

from typing import List, Tuple, Literal
import numpy as np
from qiskit.quantum_info import Clifford, Pauli


def gf2_rank(mat: np.ndarray) -> int:
    """Rank over GF(2) via Gaussian elimination."""
    A = (mat.copy() & 1).astype(np.uint8)
    m, n = A.shape
    r = 0
    c = 0
    for i in range(m):
        while c < n:
            piv = None
            for rr in range(i, m):
                if A[rr, c]:
                    piv = rr
                    break
            if piv is None:
                c += 1
                continue
            if piv != i:
                A[[i, piv]] = A[[piv, i]]
            for rr in range(i + 1, m):
                if A[rr, c]:
                    A[rr] ^= A[i]
            r += 1
            c += 1
            break
        if c >= n:
            break
    return r


def pauli_to_xz(p: Pauli, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x,z) as uint8 vectors length n."""
    z = np.array(p.z, dtype=np.uint8)
    x = np.array(p.x, dtype=np.uint8)
    if len(x) != n or len(z) != n:
        raise ValueError("Pauli size mismatch.")
    return x, z


def make_Z_on_qubit(n: int, q: int) -> Pauli:
    """Pauli Z on qubit q (using x/z boolean arrays)."""
    z = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=bool)
    z[q] = True
    return Pauli((z, x))


ConjDir = Literal["U_P_Udg", "Udg_P_U"]
IndexMap = Literal["direct", "reversed"]


def stabilizers_from_ancilla_Z_images(
    qc,
    *,
    conj: ConjDir,
    index_map: IndexMap,
) -> np.ndarray:
    """
    Build rows for images of ancilla stabilizers Z_i under the encoder.

    Returns matrix M of shape (m, 2n) where each row is (x|z).
    """
    cliff = Clifford(qc)
    cliff_inv = cliff.adjoint()  # inverse for Clifford

    n = qc.num_qubits
    m = qc.num_qubits - (qc.num_qubits - (qc.num_qubits - 0))  # unused; kept simple below


    rows = []

    # We'll infer ancilla count later; caller uses full n-k.
    # Here we just provide a helper, so caller will slice i range externally.
    # (kept for clarity)
    return rows


def build_image_rows(
    qc,
    *,
    num_ancillas: int,
    conj: ConjDir,
    index_map: IndexMap,
) -> np.ndarray:
    """
    Construct the (x|z) rows for { U Z_i U† } or { U† Z_i U } for ancilla qubits.
    Also supports direct vs reversed qubit indexing mapping.
    """
    cliff = Clifford(qc)
    cliff_inv = cliff.adjoint()  # for Clifford, adjoint == inverse

    n = qc.num_qubits
    rows: List[np.ndarray] = []

    for i in range(num_ancillas):
        q = i if index_map == "direct" else (n - 1 - i)
        P = make_Z_on_qubit(n, q)

        if conj == "U_P_Udg":
            # want: U P U†
            # Qiskit’s evolve() convention can be tricky; so we explicitly do:
            # P' = P.evolve(Clifford(U))  OR P.evolve(Clifford(U)^{-1})
            # We will test both conventions by using conj variants at a higher level.
            P2 = P.evolve(cliff)  # candidate for U P U†
        else:
            # want: U† P U
            P2 = P.evolve(cliff_inv)

        x2, z2 = pauli_to_xz(P2, n)
        rows.append(np.concatenate([x2, z2]).astype(np.uint8) & 1)

    return (np.vstack(rows) & 1).astype(np.uint8)


def verify_stabilizer_span_algorithm1(Hs: List[List[int]], qc) -> None:
    """
    Robust stabilizer-span check for Algorithm 1 encoder circuits.

    It tries 4 combinations:
      - conjugation direction: U P U†  vs  U† P U
      - index mapping: direct vs reversed
    and reports which one matches rowspace(Hs).
    """
    H = (np.array(Hs, dtype=np.uint8) & 1)
    m, two_n = H.shape
    n = two_n // 2
    k = n - m
    num_ancillas = n - k  # = m for standard stabilizer check matrix

    target_rank = gf2_rank(H)

    print(f"\n[Verifier v2] n={n}, k={k}, m={m} (ancillas={num_ancillas})")
    print(f"rank(rowspace(Hs)) = {target_rank}")

    best = None

    for conj in ("U_P_Udg", "Udg_P_U"):
        for index_map in ("direct", "reversed"):
            enc = build_image_rows(
                qc,
                num_ancillas=num_ancillas,
                conj=conj,           # type: ignore[arg-type]
                index_map=index_map, # type: ignore[arg-type]
            )

            enc_rank = gf2_rank(enc)
            stacked_rank = gf2_rank(np.vstack([H, enc]) & 1)

            ok = (enc_rank == target_rank) and (stacked_rank == target_rank)

            print(
                f"  - conj={conj:8s}, index={index_map:8s} | "
                f"rank(images)={enc_rank}, rank(stacked)={stacked_rank}  "
                f"{'✅ MATCH' if ok else '❌'}"
            )

            if ok:
                best = (conj, index_map)

    if best is None:
        print("\nResult: ❌ No combination matched.")
        print("This usually means one of:")
        print("  1) The circuit is not the intended encoder for this Hs (pivot/control assumptions broken), or")
        print("  2) Hs is not in the exact standard form assumed by Algorithm 1 (I_r in X-part), or")
        print("  3) The verification still needs adjustment to your specific qubit placement convention.")
    else:
        print(f"\nResult: ✅ Stabilizer span matches Hs using conj={best[0]} and index_map={best[1]}.")