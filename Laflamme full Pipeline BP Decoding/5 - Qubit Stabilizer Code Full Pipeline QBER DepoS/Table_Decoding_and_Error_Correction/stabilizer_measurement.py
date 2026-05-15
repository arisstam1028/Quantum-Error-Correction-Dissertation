# Purpose:
#   Converts Pauli stabilizer generators into binary symplectic checks
#   and computes five-qubit code syndromes.
#
# Process:
#   1. Translate each stabilizer string into X and Z binary rows.
#   2. Store Hx and Hz for later construction of the BP parity check.
#   3. Compute syndrome bits from the symplectic commutation product.
#
# Theory link:
#   Stabilizer measurements identify violated parity constraints without
#   observing the logical state. The binary symplectic product implements
#   the Pauli commutation rule used in syndrome extraction.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class BinarySymplectic:
    @staticmethod
    def pauli_char_to_bits(pauli: str) -> tuple[int, int]:
        """
        Convert one Pauli character to binary symplectic coordinates.

        Role in pipeline:
            Defines the common representation used by the channel,
            syndrome calculator, and BP decoder wrapper.
        """
        if pauli == "I":
            return 0, 0
        if pauli == "X":
            return 1, 0
        if pauli == "Z":
            return 0, 1
        if pauli == "Y":
            return 1, 1
        raise ValueError(f"Invalid Pauli symbol: {pauli}")

    @classmethod
    def pauli_string_to_bsf(cls, pauli_string: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a stabilizer Pauli string into X and Z binary vectors.

        Role in pipeline:
            Builds the Hx and Hz matrices used to calculate syndromes.
        """
        ex = np.zeros(len(pauli_string), dtype=np.uint8)
        ez = np.zeros(len(pauli_string), dtype=np.uint8)

        for i, p in enumerate(pauli_string):
            x, z = cls.pauli_char_to_bits(p)
            ex[i] = x
            ez[i] = z

        return ex, ez


@dataclass(frozen=True)
class StabilizerMeasurement:
    stabilizers: Sequence[str]

    def __post_init__(self) -> None:
        if not self.stabilizers:
            raise ValueError("stabilizers must not be empty")

        n_qubits = len(self.stabilizers[0])
        hx_rows = []
        hz_rows = []

        for stabilizer in self.stabilizers:
            self.validate_pauli_string(stabilizer, expected_length=n_qubits)
            hx, hz = BinarySymplectic.pauli_string_to_bsf(stabilizer)
            hx_rows.append(hx)
            hz_rows.append(hz)

        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "hx", np.stack(hx_rows, axis=0).astype(np.uint8))
        object.__setattr__(self, "hz", np.stack(hz_rows, axis=0).astype(np.uint8))
        object.__setattr__(self, "n_stabilizers", len(self.stabilizers))

    def compute_syndrome(self, ex: np.ndarray, ez: np.ndarray) -> str:
        """
        Compute the stabilizer syndrome for a binary symplectic error.

        Role in pipeline:
            Produces the parity constraints that BP must satisfy when
            estimating the underlying binary error pattern.
        """
        self.validate_binary_error(ex, ez, self.n_qubits)

        # syndrome_i  hx_i · ez + hz_i · ex mod 2
        syndrome_bits = (self.hx @ ez + self.hz @ ex) % 2
        return "".join(str(int(bit)) for bit in syndrome_bits)

    @staticmethod
    def validate_binary_error(
        ex: np.ndarray,
        ez: np.ndarray,
        expected_length: int | None = None,
    ) -> None:
        if ex.ndim != 1 or ez.ndim != 1:
            raise ValueError("ex and ez must be 1D arrays")

        if len(ex) != len(ez):
            raise ValueError("ex and ez must have the same length")

        if expected_length is not None and len(ex) != expected_length:
            raise ValueError(
                f"Expected binary error length {expected_length}, got {len(ex)}"
            )

        if not np.all((ex == 0) | (ex == 1)):
            raise ValueError("ex must contain only 0/1 values")

        if not np.all((ez == 0) | (ez == 1)):
            raise ValueError("ez must contain only 0/1 values")

    @classmethod
    def validate_pauli_string(
        cls,
        pauli: str,
        expected_length: int | None = None,
    ) -> None:
        if expected_length is not None and len(pauli) != expected_length:
            raise ValueError(
                f"Expected Pauli string of length {expected_length}, got {len(pauli)}"
            )

        invalid = [symbol for symbol in pauli if symbol not in ("I", "X", "Y", "Z")]
        if invalid:
            raise ValueError(f"Invalid Pauli symbols in {pauli!r}: {invalid}")
