from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from verify_encoder_v2 import verify_stabilizer_span_algorithm1


def remove_gate_by_index(qc: QuantumCircuit, idx: int) -> QuantumCircuit:
    """Return a new circuit with the idx-th instruction removed."""
    new_qc = QuantumCircuit(qc.num_qubits, name=qc.name)
    for i, (inst, qargs, cargs) in enumerate(qc.data):
        if i == idx:
            continue
        new_qc.append(inst, qargs, cargs)
    return new_qc


def find_removable_cz_gates(Hs, qc: QuantumCircuit) -> List[int]:
    """
    Brute-force (but small-circuit friendly) check:
    A CZ is 'removable' if deleting it preserves the stabilizer span match.
    """
    removable = []
    cz_indices = [i for i, (inst, _, __) in enumerate(qc.data) if inst.name == "cz"]

    # First confirm baseline circuit passes
    print("\nBaseline check:")
    verify_stabilizer_span_algorithm1(Hs, qc)

    for idx in cz_indices:
        test = remove_gate_by_index(qc, idx)

        # We only care if it STILL matches
        # We'll detect match by capturing stdout? Simple approach:
        # re-run verifier manually in your terminal and look for MATCH
        # But better: implement a boolean verifier. For now we do a lightweight version:
        ok = _stabilizer_span_matches(Hs, test)

        if ok:
            removable.append(idx)

    return removable


def _stabilizer_span_matches(Hs, qc: QuantumCircuit) -> bool:
    """
    Boolean version of the successful condition from verifier v2:
    conj=Udg_P_U and index_map=direct
    """
    import numpy as np
    from qiskit.quantum_info import Clifford, Pauli

    H = (np.array(Hs, dtype=np.uint8) & 1)
    m, two_n = H.shape
    n = two_n // 2
    num_ancillas = m

    def gf2_rank(mat: np.ndarray) -> int:
        A = (mat.copy() & 1).astype(np.uint8)
        mm, nn = A.shape
        r = 0
        c = 0
        for i in range(mm):
            while c < nn:
                piv = None
                for rr in range(i, mm):
                    if A[rr, c]:
                        piv = rr
                        break
                if piv is None:
                    c += 1
                    continue
                if piv != i:
                    A[[i, piv]] = A[[piv, i]]
                for rr in range(i + 1, mm):
                    if A[rr, c]:
                        A[rr] ^= A[i]
                r += 1
                c += 1
                break
            if c >= nn:
                break
        return r

    cliff = Clifford(qc)
    cliff_inv = cliff.adjoint()

    rows = []
    for i in range(num_ancillas):
        z = np.zeros(n, dtype=bool)
        x = np.zeros(n, dtype=bool)
        z[i] = True
        P = Pauli((z, x))
        P2 = P.evolve(cliff_inv)  # U† P U (the matching convention you found)
        z2 = np.array(P2.z, dtype=np.uint8)
        x2 = np.array(P2.x, dtype=np.uint8)
        rows.append(np.concatenate([x2, z2]) & 1)

    enc = (np.vstack(rows) & 1).astype(np.uint8)

    target_rank = gf2_rank(H)
    enc_rank = gf2_rank(enc)
    stacked_rank = gf2_rank(np.vstack([H, enc]) & 1)

    return (enc_rank == target_rank) and (stacked_rank == target_rank)


def build_simplified_by_removing_indices(qc: QuantumCircuit, remove_idxs: List[int]) -> QuantumCircuit:
    """Remove a list of indices (assumed sorted ascending) from qc."""
    remove_set = set(remove_idxs)
    new_qc = QuantumCircuit(qc.num_qubits, name=qc.name + "_simplified")
    for i, (inst, qargs, cargs) in enumerate(qc.data):
        if i in remove_set:
            continue
        new_qc.append(inst, qargs, cargs)
    return new_qc