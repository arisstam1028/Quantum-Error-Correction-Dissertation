# Algorithm1.py
# Implementation of Algorithm 1 (encoder construction) with *native* CY/CZ gates
# so the drawn circuit matches the paper more closely.
#
# Requires: qiskit, matplotlib
# Usage: import from main.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

from qiskit import QuantumCircuit
from qiskit.circuit.library import CYGate, CZGate


BinaryVec = Sequence[int]
BinaryMat = Sequence[Sequence[int]]


# Utility Functions

def _validate_binary_matrix(Hs: BinaryMat) -> None:
    if not Hs or not Hs[0]:
        raise ValueError("Hs must be a non-empty matrix.")
    row_len = len(Hs[0])
    if row_len % 2 != 0:
        raise ValueError(f"Hs must have 2n columns; got {row_len} columns (odd).")
    for i, row in enumerate(Hs):
        if len(row) != row_len:
            raise ValueError(f"Row {i} length {len(row)} != {row_len}.")
        if any(b not in (0, 1) for b in row):
            raise ValueError(f"Row {i} contains non-binary entries.")


def _validate_binary_vec(v: BinaryVec, length: int, name: str) -> None:
    if len(v) != length:
        raise ValueError(f"{name} must have length {length}, got {len(v)}.")
    if any(b not in (0, 1) for b in v):
        raise ValueError(f"{name} contains non-binary entries.")


def _gf2_rank(mat: List[List[int]]) -> int:
    """Compute rank over GF(2) via Gaussian elimination."""
    if not mat:
        return 0
    A = [row[:] for row in mat]
    m, n = len(A), len(A[0])
    rank = 0
    col = 0

    for r in range(m):
        while col < n:
            pivot = None
            for rr in range(r, m):
                if A[rr][col] == 1:
                    pivot = rr
                    break
            if pivot is None:
                col += 1
                continue

            A[r], A[pivot] = A[pivot], A[r]

            for rr in range(r + 1, m):
                if A[rr][col] == 1:
                    A[rr] = [a ^ b for a, b in zip(A[rr], A[r])]

            rank += 1
            col += 1
            break

        if col >= n:
            break

    return rank


def _apply_cy_native(qc: QuantumCircuit, ctrl: int, tgt: int) -> None:
    """Use Qiskit's native CY gate so it draws as CY (not decomposed)."""
    qc.append(CYGate(), [ctrl, tgt])


def _apply_cz_native(qc: QuantumCircuit, ctrl: int, tgt: int) -> None:
    """
    Use Qiskit's CZ gate object (still draws as the symmetric ●—● CZ).
    If you want a custom label, you can set CZGate(label"CZ") etc.
    """
    qc.append(CZGate(), [ctrl, tgt])


# Encoder Class

@dataclass(frozen=True)
class EncoderSpec:
    n: int
    k: int
    r: int
    num_stabilizers: int


class StabilizerEncoder:
    """
    Encoder builder from standard-form stabilizer matrix Hs  [X | Z].

    Automatic inference:
      n  (#cols)/2
      k  n - (#rows)         (assuming Hs has (n-k) rows)
      r  rank over GF(2) of X-part (first n columns)

    Gate representation:
      - CY uses native CYGate (draws as CY)
      - CZ uses native CZGate (draws as ●—●)
    """

    def __init__(
        self,
        Hs: BinaryMat,
        logical_X: Union[BinaryVec, Sequence[BinaryVec]],
        name: str = "encoder",
        *,
        cy_as_native: bool = True,
        cz_as_native: bool = True,
    ):
        _validate_binary_matrix(Hs)
        self.Hs: List[List[int]] = [list(row) for row in Hs]
        self.name = name

        self.n = len(self.Hs[0]) // 2
        self.num_stabilizers = len(self.Hs)
        self.k = self.n - self.num_stabilizers
        if self.k < 0:
            raise ValueError(
                f"Inferred k is negative (k = n - #rows = {self.n} - {self.num_stabilizers} = {self.k}). "
                "Hs does not look like an (n-k) x 2n check matrix."
            )

        X_part = [row[: self.n] for row in self.Hs]
        self.r = _gf2_rank(X_part)

        total_len = 2 * self.n
        if self.k == 0:
            self.logical_Xs: List[List[int]] = []
        else:
            # normalize to list of vectors
            if isinstance(logical_X[0], (list, tuple)):  # type: ignore[index]
                logical_list = list(logical_X)  # type: ignore[assignment]
            else:
                logical_list = [logical_X]  # type: ignore[list-item]

            if len(logical_list) != self.k:
                raise ValueError(
                    f"Inferred k={self.k}, but you provided {len(logical_list)} logical_X vector(s). "
                    "Provide exactly k vectors (each length 2n), or use a code with k=1."
                )

            self.logical_Xs = []
            for idx, v in enumerate(logical_list):
                _validate_binary_vec(v, total_len, f"logical_X[{idx}]")
                self.logical_Xs.append(list(v))

        self._cy_as_native = cy_as_native
        self._cz_as_native = cz_as_native

    @property
    def spec(self) -> EncoderSpec:
        return EncoderSpec(n=self.n, k=self.k, r=self.r, num_stabilizers=self.num_stabilizers)

    def _apply_pair(self, qc: QuantumCircuit, ctrl: int, tgt: int, xbit: int, zbit: int) -> None:
        """Apply the appropriate controlled Pauli based on (x,z) pair."""
        pair = (xbit, zbit)
        if pair == (1, 0):
            qc.cx(ctrl, tgt)
        elif pair == (0, 1):
            if self._cz_as_native:
                _apply_cz_native(qc, ctrl, tgt)
            else:
                qc.cz(ctrl, tgt)
        elif pair == (1, 1):
            if self._cy_as_native:
                _apply_cy_native(qc, ctrl, tgt)
            else:
                qc.cy(ctrl, tgt)
        # else (0,0): nothing

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name=self.name)

        #  Step A: encode logical X (controlled by message/logical input qubit(s)) 
        # logical qubits assumed at indices (n-k ... n-1)
        for i in range(self.k):
            msg = self.n - self.k + i
            xL = self.logical_Xs[i][: self.n]
            zL = self.logical_Xs[i][self.n :]

            for j in range(self.n):
                if j == msg:
                    continue
                self._apply_pair(qc, msg, j, xL[j], zL[j])

        #  Step B: stabilizer loop (Algorithm 1 uses i1..r) 
        for i in range(self.r):
            ctrl = i

            # Prepare pivot/control qubit: H, and if z_diag1 then S after H
            z_diag = self.Hs[i][self.n + i]
            qc.h(ctrl)
            if z_diag == 1:
                qc.s(ctrl)

            # Controlled operations to other qubits from stabilizer row i
            for j in range(self.n):
                if j == ctrl:
                    continue
                xbit = self.Hs[i][j]
                zbit = self.Hs[i][self.n + j]
                self._apply_pair(qc, ctrl, j, xbit, zbit)

        return qc


# Printing Class

class HsPrinter:
    @staticmethod
    def _format_vec(v: BinaryVec) -> str:
        n2 = len(v)
        if n2 % 2 != 0:
            return "".join(str(b) for b in v)
        n = n2 // 2
        x = "".join(str(b) for b in v[:n])
        z = "".join(str(b) for b in v[n:])
        return f"{x} | {z}"

    @staticmethod
    def print_all(encoder: StabilizerEncoder, logical_label: str = "logical_X") -> None:
        spec = encoder.spec
        print(f'n  {spec.n}, k  {spec.k}, r  {spec.r}, #stabilizers  {spec.num_stabilizers}\n')

        print("Hs (X | Z):")
        for row in encoder.Hs:
            print(HsPrinter._format_vec(row))

        if spec.k == 0:
            print('\n(No logical qubits: k0.)')
            return

        print(f"\n{logical_label} vector(s):")
        for i, v in enumerate(encoder.logical_Xs):
            print(f'  {logical_label}[{i}]  {HsPrinter._format_vec(v)}')


# Plotting Class

class CircuitPlotter:
    @staticmethod
    def show(qc: QuantumCircuit, *, fold: int = 80, block: bool = True) -> None:
        import matplotlib.pyplot as plt
        qc.draw(output="mpl", fold=fold)
        plt.show(block=block)