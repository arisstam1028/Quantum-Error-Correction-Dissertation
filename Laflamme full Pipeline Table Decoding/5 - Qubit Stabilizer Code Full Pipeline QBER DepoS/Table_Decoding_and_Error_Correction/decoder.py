# Purpose:
#   Decode five-qubit syndromes using a single-qubit lookup table.
#
# Process:
#   1. Enumerate identity and all single-qubit X, Y, Z errors.
#   2. Compute their syndromes with the stabilizer measurement object.
#   3. Return the stored correction matching an observed syndrome.
#
# Theory link:
#   The five-qubit code corrects any single-qubit Pauli error. Table
#   decoding maps each syndrome to a representative Pauli correction.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stabilizer_measurement import StabilizerMeasurement

@dataclass(frozen=True)
class SyndromeTableDecoder:
    stabilizer_measurement: StabilizerMeasurement

    def __post_init__(self) -> None:
        object.__setattr__(self, "lookup", self._build_lookup())

    def _build_lookup(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Hard-decision decoder:
        Each syndrome is mapped to ONE fixed correction.

        Built natively in binary symplectic form.

        Role in pipeline:
            Precomputes the syndrome-to-correction map used during
            Monte Carlo simulation.
        """
        n_qubits = self.stabilizer_measurement.n_qubits
        lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Identity
        ex0 = np.zeros(n_qubits, dtype=np.uint8)
        ez0 = np.zeros(n_qubits, dtype=np.uint8)
        syndrome0 = self.stabilizer_measurement.compute_syndrome(ex0, ez0)
        lookup[syndrome0] = (ex0.copy(), ez0.copy())

        # All single-qubit X, Y, Z errors
        for qubit in range(n_qubits):
            for pauli in ("X", "Y", "Z"):
                ex, ez = self.make_single_qubit_error(pauli, qubit, n_qubits)
                syndrome = self.stabilizer_measurement.compute_syndrome(ex, ez)

                if syndrome not in lookup:
                    lookup[syndrome] = (ex.copy(), ez.copy())

        return lookup

    def decode(self, syndrome: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return fixed correction for this syndrome.

        Role in pipeline:
            Converts a syndrome into correction vectors (cx, cz), which
            are combined with the actual error to form a residual.
        """
        if syndrome in self.lookup:
            ex, ez = self.lookup[syndrome]
            return ex.copy(), ez.copy()

        n_qubits = self.stabilizer_measurement.n_qubits
        return (
            np.zeros(n_qubits, dtype=np.uint8),
            np.zeros(n_qubits, dtype=np.uint8),
        )

    @staticmethod
    def make_single_qubit_error(
        pauli: str,
        qubit: int,
        n_qubits: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid Pauli {pauli}")

        ex = np.zeros(n_qubits, dtype=np.uint8)
        ez = np.zeros(n_qubits, dtype=np.uint8)

        if pauli == "X":
            ex[qubit] = 1
        elif pauli == "Z":
            ez[qubit] = 1
        else:  # Y
            ex[qubit] = 1
            ez[qubit] = 1

        return ex, ez
