from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BasisEncoder:
    """
    Basis encoder: classical bits -> computational basis states.

    Mapping:
      b=0 -> |0>  (do nothing, assuming qubits start in |0>)
      b=1 -> |1>  (apply X)

    Mathematically: |b> = X^b |0>
    """

    def encode_bits(self, qc: QuantumCircuit, qubits: Sequence[int], bits: Sequence[int]) -> None:
        if len(qubits) != len(bits):
            raise ValueError(f"Length mismatch: qubits={len(qubits)} bits={len(bits)}")

        for i, (q, b) in enumerate(zip(qubits, bits)):
            if b not in (0, 1):
                raise ValueError(f"bits[{i}]={b} is not 0 or 1")
            if b == 1:
                qc.x(q)

    def encode_int(
        self,
        qc: QuantumCircuit,
        qubits: Sequence[int],
        value: int,
        *,
        little_endian: bool = True,
    ) -> List[int]:
        n = len(qubits)
        if value < 0 or value >= (1 << n):
            raise ValueError(f"value={value} out of range for {n} qubits (0..{(1<<n)-1})")

        bits = [(value >> i) & 1 for i in range(n)]
        if not little_endian:
            bits = bits[::-1]

        self.encode_bits(qc, qubits, bits)
        return bits

    # -------------------------
    # Printing helpers (as requested)
    # -------------------------
    def print_encoding_summary_bits(self, bits: Sequence[int], qubits: Sequence[int] | None = None) -> None:
        if qubits is None:
            qubits = list(range(len(bits)))
        print("=== Basis Encoding Summary ===")
        print(f"Target qubits: {list(qubits)}")
        print(f"Input bits (q0..): {list(bits)}")
        ones = [q for q, b in zip(qubits, bits) if b == 1]
        print(f"Applied X on qubits: {ones if ones else 'none'}")

    def print_encoding_summary_int(
        self,
        value: int,
        bits_used: Sequence[int],
        *,
        little_endian: bool,
        qubits: Sequence[int] | None = None,
    ) -> None:
        if qubits is None:
            qubits = list(range(len(bits_used)))
        print("=== Basis Encoding Summary ===")
        print(f"Target qubits: {list(qubits)}")
        print(f"Input integer: {value}")
        print(f"Endian mode: {'little' if little_endian else 'big'}")
        print(f"Bits used (q0..): {list(bits_used)}")
        ones = [q for q, b in zip(qubits, bits_used) if b == 1]
        print(f"Applied X on qubits: {ones if ones else 'none'}")


@dataclass(frozen=True)
class CircuitPlotter:
    """
    Responsible only for plotting circuits (as requested).
    """

    def show(self, qc: QuantumCircuit, title: Optional[str] = None) -> None:
        qc.draw(output="mpl")
        if title:
            plt.title(title)
        plt.show()

    def show_without_measurements(self, qc: QuantumCircuit, title: Optional[str] = None) -> None:
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        qc_no_meas.draw(output="mpl")
        if title:
            plt.title(title)
        plt.show()