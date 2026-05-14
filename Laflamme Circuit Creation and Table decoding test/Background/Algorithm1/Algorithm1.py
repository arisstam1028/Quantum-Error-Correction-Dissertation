# Algorithm1.py
# Implementation of Algorithm 1 from
# "Quantum Circuits for Stabilizer Error-Correcting Codes: A Tutorial"

from dataclasses import dataclass
from typing import List, Sequence, Union
from qiskit import QuantumCircuit


# ==========================
# Utility Functions
# ==========================

def _gf2_rank(matrix: List[List[int]]) -> int:
    """Compute rank over GF(2) using Gaussian elimination."""
    A = [row[:] for row in matrix]
    m = len(A)
    n = len(A[0])
    rank = 0
    col = 0

    for r in range(m):
        while col < n:
            pivot = None
            for i in range(r, m):
                if A[i][col] == 1:
                    pivot = i
                    break
            if pivot is None:
                col += 1
                continue

            A[r], A[pivot] = A[pivot], A[r]

            for i in range(r + 1, m):
                if A[i][col] == 1:
                    A[i] = [a ^ b for a, b in zip(A[i], A[r])]

            rank += 1
            col += 1
            break

        if col >= n:
            break

    return rank


def _apply_cy(qc: QuantumCircuit, ctrl: int, tgt: int):
    """Controlled-Y decomposition."""
    qc.s(tgt)
    qc.cx(ctrl, tgt)
    qc.sdg(tgt)


# ==========================
# Encoder Class
# ==========================

@dataclass
class EncoderSpec:
    n: int
    k: int
    r: int
    stabilizers: int


class StabilizerEncoder:
    """
    Encoder builder based on Algorithm 1.
    Automatically infers n, k and r from Hs.
    """

    def __init__(self, Hs: Sequence[Sequence[int]],
                 logical_X: Union[Sequence[int], Sequence[Sequence[int]]],
                 name: str = "encoder"):

        self.Hs = [list(row) for row in Hs]
        self.n = len(self.Hs[0]) // 2
        self.stabilizers = len(self.Hs)
        self.k = self.n - self.stabilizers

        # rank of X part
        X_part = [row[:self.n] for row in self.Hs]
        self.r = _gf2_rank(X_part)

        # normalise logical_X
        if self.k == 1:
            self.logical_X = [list(logical_X)]
        else:
            self.logical_X = [list(v) for v in logical_X]

        self.name = name

    @property
    def spec(self):
        return EncoderSpec(self.n, self.k, self.r, self.stabilizers)

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name=self.name)

        # -------------------------
        # Step A: Logical X encoding
        # -------------------------
        for i in range(self.k):
            msg = self.n - self.k + i
            xL = self.logical_X[i][:self.n]
            zL = self.logical_X[i][self.n:]

            for j in range(self.n):
                if j == msg:
                    continue
                pair = (xL[j], zL[j])
                if pair == (1, 0):
                    qc.cx(msg, j)
                elif pair == (0, 1):
                    qc.cz(msg, j)
                elif pair == (1, 1):
                    _apply_cy(qc, msg, j)

        # -------------------------
        # Step B: Stabilizer section
        # -------------------------
        for i in range(self.r):
            ctrl = i

            qc.h(ctrl)

            if self.Hs[i][self.n + i] == 1:
                qc.s(ctrl)

            for j in range(self.n):
                if j == ctrl:
                    continue

                xbit = self.Hs[i][j]
                zbit = self.Hs[i][self.n + j]
                pair = (xbit, zbit)

                if pair == (1, 0):
                    qc.cx(ctrl, j)
                elif pair == (0, 1):
                    qc.cz(ctrl, j)
                elif pair == (1, 1):
                    _apply_cy(qc, ctrl, j)

        return qc


# ==========================
# Printing Class
# ==========================

class HsPrinter:

    @staticmethod
    def print_all(encoder: StabilizerEncoder):
        spec = encoder.spec
        print(f"n = {spec.n}")
        print(f"k = {spec.k}")
        print(f"r = {spec.r}")
        print(f"Number of stabilizers = {spec.stabilizers}\n")

        print("Hs (X | Z):")
        for row in encoder.Hs:
            x = ''.join(str(b) for b in row[:spec.n])
            z = ''.join(str(b) for b in row[spec.n:])
            print(f"{x} | {z}")

        print("\nLogical X:")
        for vec in encoder.logical_X:
            x = ''.join(str(b) for b in vec[:spec.n])
            z = ''.join(str(b) for b in vec[spec.n:])
            print(f"{x} | {z}")


# ==========================
# Plotting Class
# ==========================

class CircuitPlotter:

    @staticmethod
    def show(qc: QuantumCircuit):
        import matplotlib.pyplot as plt
        qc.draw("mpl")
        plt.show()