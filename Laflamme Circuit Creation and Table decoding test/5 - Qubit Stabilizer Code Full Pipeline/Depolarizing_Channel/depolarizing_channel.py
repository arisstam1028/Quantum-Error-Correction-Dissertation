from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from .pauli import (
    AppliedError,
    apply_pattern_to_circuit,
    format_pattern,
    make_rng,
    sample_error_pattern,
    validate_qubits,
)


class DepolarizingChannel:
    """
    General independent single-qubit depolarizing channel.

    For each target qubit:
        I with probability 1 - p
        X with probability p / 3
        Y with probability p / 3
        Z with probability p / 3
    """

    def __init__(self, p: float, seed: Optional[int] = None) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must satisfy 0 <= p <= 1, got {p}")

        self.p = float(p)
        self.rng = make_rng(seed)

    def sample_error_pattern(self, qubits: Sequence[int]) -> List[str]:
        return sample_error_pattern(qubits, self.p, self.rng)

    def apply_pattern(
        self,
        qc: QuantumCircuit,
        qubits: Sequence[int],
        pattern: Sequence[str],
        *,
        inplace: bool = False,
        barrier: bool = False,
    ) -> Tuple[QuantumCircuit, List[AppliedError]]:
        validate_qubits(qubits)

        out = qc if inplace else qc.copy()

        if barrier:
            out.barrier()

        applied = apply_pattern_to_circuit(out, qubits, pattern)

        if barrier:
            out.barrier()

        return out, applied

    def apply_random(
        self,
        qc: QuantumCircuit,
        qubits: Optional[Sequence[int]] = None,
        *,
        inplace: bool = False,
        barrier: bool = False,
    ) -> Tuple[QuantumCircuit, List[str], List[AppliedError]]:
        if qubits is None:
            qubits = list(range(qc.num_qubits))

        pattern = self.sample_error_pattern(qubits)
        out, applied = self.apply_pattern(
            qc=qc,
            qubits=qubits,
            pattern=pattern,
            inplace=inplace,
            barrier=barrier,
        )
        return out, pattern, applied

    def apply_manual(
        self,
        qc: QuantumCircuit,
        qubits: Sequence[int],
        pattern: Sequence[str],
        *,
        inplace: bool = False,
        barrier: bool = False,
    ) -> Tuple[QuantumCircuit, List[AppliedError]]:
        return self.apply_pattern(
            qc=qc,
            qubits=qubits,
            pattern=pattern,
            inplace=inplace,
            barrier=barrier,
        )

    @staticmethod
    def format_pattern(qubits: Sequence[int], pattern: Sequence[str]) -> str:
        return format_pattern(qubits, pattern)