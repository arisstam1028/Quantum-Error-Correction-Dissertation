# simplified_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qiskit import QuantumCircuit, transpile

# Uses your proven-correct verifier convention:
#   match iff conj=Udg_P_U and index_map=direct
from verify_encoder_v2 import verify_stabilizer_span_algorithm1
from verify_encoder_v2 import gf2_rank  # if you didn't export it, I'll inline below if needed

import numpy as np
from qiskit.quantum_info import Clifford, Pauli


@dataclass(frozen=True)
class SimplifySpec:
    optimization_level: int = 3
    basis_gates: tuple[str, ...] = ("h", "s", "sdg", "cx", "cz", "cy")


def _stabilizer_span_matches(Hs, qc: QuantumCircuit) -> bool:
    """
    Boolean version of the successful condition from your Verifier v2:
    - conj=Udg_P_U
    - index_map=direct
    """
    H = (np.array(Hs, dtype=np.uint8) & 1)
    m, two_n = H.shape
    n = two_n // 2
    num_ancillas = m

    def _gf2_rank_local(mat: np.ndarray) -> int:
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
        P2 = P.evolve(cliff_inv)  # U† P U (the matching convention)
        z2 = np.array(P2.z, dtype=np.uint8)
        x2 = np.array(P2.x, dtype=np.uint8)
        rows.append(np.concatenate([x2, z2]) & 1)

    enc = (np.vstack(rows) & 1).astype(np.uint8)

    target_rank = _gf2_rank_local(H)
    enc_rank = _gf2_rank_local(enc)
    stacked_rank = _gf2_rank_local(np.vstack([H, enc]) & 1)

    return (enc_rank == target_rank) and (stacked_rank == target_rank)


def _remove_gate_indices(qc: QuantumCircuit, remove_idxs: set[int]) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits, name=(qc.name + "_pruned" if qc.name else "pruned"))
    for i, (inst, qargs, cargs) in enumerate(qc.data):
        if i in remove_idxs:
            continue
        out.append(inst, qargs, cargs)
    return out


class EncoderSimplifier:
    """
    Simplify an existing encoder circuit.

    Two-stage approach:
      1) Qiskit optimization (may do nothing)
      2) Paper-style semantic pruning: remove CZ gates that do not change the stabilizer code
         (verified by stabilizer-span equivalence).
    """

    def __init__(
        self,
        Hs,
        *,
        optimization_level: int = 3,
        basis_gates: tuple[str, ...] = ("h", "s", "sdg", "cx", "cz", "cy"),
        do_semantic_cz_prune: bool = True,
    ):
        self.Hs = Hs
        self.spec = SimplifySpec(optimization_level=optimization_level, basis_gates=basis_gates)
        self.do_semantic_cz_prune = do_semantic_cz_prune

    def simplify(self, qc: QuantumCircuit, *, new_name: str | None = None) -> QuantumCircuit:
        # Stage 1: compiler opt
        opt = transpile(
            qc,
            basis_gates=list(self.spec.basis_gates),
            optimization_level=self.spec.optimization_level,
        )
        opt.name = new_name or (qc.name + "_opt" if qc.name else "opt")

        if not self.do_semantic_cz_prune:
            return opt

        # Stage 2: semantic CZ pruning
        # Baseline must match
        if not _stabilizer_span_matches(self.Hs, opt):
            raise ValueError("Baseline optimized circuit does not match Hs stabilizer span. Refusing to prune.")

        cz_indices = [i for i, (inst, _, __) in enumerate(opt.data) if inst.name == "cz"]
        remove = set()

        # Greedy pass: try removing each CZ in order; keep it removed if still matches.
        current = opt
        for idx in cz_indices:
            # Because indices shift after removals, we re-find the idx-th CZ each time by position.
            # Simpler: operate on instruction list directly using a fresh scan each iteration.
            cz_positions = [i for i, (inst, _, __) in enumerate(current.data) if inst.name == "cz"]
            if not cz_positions:
                break
            # Map "original order" attempt: take the first remaining CZ each loop.
            trial_pos = cz_positions[0]

            trial = _remove_gate_indices(current, {trial_pos})
            if _stabilizer_span_matches(self.Hs, trial):
                current = trial  # keep removal
            else:
                # If not removable, skip it by moving it past: remove nothing, but delete it from future attempts.
                # Implement by temporarily marking it with a no-op approach: easiest is to rotate list:
                # We'll just move to next by deleting it from the attempt list:
                # do nothing; but we need to prevent infinite loop:
                # So we "lock" it by replacing it with itself via a flag.
                # Easiest: pop it from consideration by removing it from current scan:
                # We'll implement by removing it from cz_positions list via a counter mechanism:
                # To keep this simple and robust, do a single pass over current.data indices:
                pass

            # To avoid the "lock" complexity, we instead do a full scan each time:
            # Try removing *any* CZ that is removable, repeat until none removable.
            # We'll break here and do that outer loop next.
            break

        # Robust prune loop: keep removing any removable CZ until fixed point
        changed = True
        current = opt
        while changed:
            changed = False
            cz_positions = [i for i, (inst, _, __) in enumerate(current.data) if inst.name == "cz"]
            for pos in cz_positions:
                trial = _remove_gate_indices(current, {pos})
                if _stabilizer_span_matches(self.Hs, trial):
                    current = trial
                    changed = True
                    break  # restart scan since indices changed

        current.name = new_name or (qc.name + "_simplified" if qc.name else "simplified")
        return current


class SimplifiedCircuitPlotter:
    @staticmethod
    def show_side_by_side(
        qc_left: QuantumCircuit,
        qc_right: QuantumCircuit,
        *,
        titles: tuple[str, str] = ("Original", "Simplified"),
        fold: int = 80,
        block: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(titles[0])
        ax1.axis("off")
        qc_left.draw(output="mpl", fold=fold, ax=ax1)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(titles[1])
        ax2.axis("off")
        qc_right.draw(output="mpl", fold=fold, ax=ax2)

        plt.tight_layout()
        plt.show(block=block)