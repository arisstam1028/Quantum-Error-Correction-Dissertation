# Purpose:
#   Implements a syndrome table decoder for the five-qubit stabilizer
#   code using single-qubit Pauli representatives.
#
# Process:
#   1. Enumerate the identity and all weight-one X, Y, Z errors.
#   2. Compute each error syndrome using the stabilizer checks.
#   3. Store one fixed correction for each syndrome and apply it.
#
# Theory link:
#   The five-qubit code has distance three, so single-qubit errors
#   have distinct syndromes. Table decoding maps a measured syndrome
#   to a Pauli correction in binary symplectic form.

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
            Precomputes the table used during Monte Carlo simulation so
            each observed syndrome can be corrected without optimization.
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
            Applies the table-decoding step by returning the Pauli
            representative associated with the measured syndrome.
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
        """
        Construct the binary symplectic vector for one physical Pauli error.

        Role in pipeline:
            Supplies the error representatives from which the lookup table
            is built for all weight-one correctable errors.
        """
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
