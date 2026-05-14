# verify_encoder.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import Clifford, Pauli

def gf2_rank(mat: np.ndarray) -> int:
    A = mat.copy() % 2
    m, n = A.shape
    r = 0
    c = 0
    for i in range(m):
        while c < n:
            piv = None
            for rr in range(i, m):
                if A[rr, c] == 1:
                    piv = rr
                    break
            if piv is None:
                c += 1
                continue
            if piv != i:
                A[[i, piv]] = A[[piv, i]]
            for rr in range(i + 1, m):
                if A[rr, c] == 1:
                    A[rr] ^= A[i]
            r += 1
            c += 1
            break
        if c >= n:
            break
    return r

def pauli_to_xz(p: Pauli, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # Qiskit Pauli stores (z, x) internally; access via .z/.x boolean arrays
    z = np.array(p.z, dtype=np.uint8)
    x = np.array(p.x, dtype=np.uint8)
    assert len(x) == n and len(z) == n
    return x, z

def evolve_xz_by_circuit_Z_ancillas(qc, n: int, num_ancillas: int) -> np.ndarray:
    """
    Build matrix whose rows are the (x|z) of U Z_i U^dagger for i in ancillas.
    """
    cliff = Clifford(qc)
    rows = []
    for i in range(num_ancillas):
        # Z on qubit i
        p = Pauli("I" * (n - i - 1) + "Z" + "I" * i)  # note string is little-endian in display, but Pauli handles it
        # safer: build by x/z arrays
        z = np.zeros(n, dtype=np.uint8); x = np.zeros(n, dtype=np.uint8)
        z[i] = 1
        p = Pauli((z.astype(bool), x.astype(bool)))

        p2 = p.evolve(cliff)  # p2 = U p U^dagger
        x2, z2 = pauli_to_xz(p2, n)
        rows.append(np.concatenate([x2, z2], axis=0))
    return np.array(rows, dtype=np.uint8) % 2

def verify_stabilizer_span(Hs: List[List[int]], qc) -> None:
    H = np.array(Hs, dtype=np.uint8) % 2
    m, two_n = H.shape
    n = two_n // 2
    k = n - m
    num_ancillas = n - k  # = m

    # Target stabilizer row space = row space of Hs over GF(2)
    target_rank = gf2_rank(H)

    # Encoded ancilla-Z images
    enc = evolve_xz_by_circuit_Z_ancillas(qc, n=n, num_ancillas=num_ancillas)
    enc_rank = gf2_rank(enc)

    # Compare spans by checking rank of stacked matrix
    stacked = np.vstack([H, enc]) % 2
    stacked_rank = gf2_rank(stacked)

    print(f"n={n}, k={k}, m={m}")
    print(f"rank(rowspace(Hs)) = {target_rank}")
    print(f"rank(rowspace(U Z_anc U†)) = {enc_rank}")
    print(f"rank(rowspace(combined)) = {stacked_rank}")

    if target_rank != enc_rank or stacked_rank != target_rank:
        print("❌ Not the same stabilizer span (or not enough independent generators).")
    else:
        print("✅ Encoded stabilizer span matches Hs (up to row operations).")