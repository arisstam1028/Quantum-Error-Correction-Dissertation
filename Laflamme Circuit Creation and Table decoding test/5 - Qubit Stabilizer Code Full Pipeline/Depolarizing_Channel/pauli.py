# Purpose:
#   Sample Pauli errors and apply them to Qiskit circuits.
#
# Process:
#   1. Validate target qubits and sampled Pauli patterns.
#   2. Sample I, X, Y, or Z according to a depolarizing probability.
#   3. Append the corresponding Pauli gates to the circuit.
#
# Theory link:
#   The symmetric depolarizing channel applies I with probability 1-p
#   and X, Y, Z each with probability p/3. In binary symplectic form,
#   Y contains both X and Z components.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import random

from qiskit import QuantumCircuit


VALID_PAULIS = {"I", "X", "Y", "Z"}


@dataclass
class AppliedError:
    """Record of a non-identity Pauli applied to a qubit."""
    qubit: int
    pauli: str


def validate_qubits(qubits: Sequence[int]) -> None:
    if len(qubits) == 0:
        raise ValueError("qubits must not be empty")
    if any((not isinstance(q, int)) or q < 0 for q in qubits):
        raise ValueError("qubits must be a sequence of non-negative integers")
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubits must not contain duplicates")


def validate_pattern(qubits: Sequence[int], pattern: Sequence[str]) -> None:
    if len(qubits) != len(pattern):
        raise ValueError("pattern length must match number of qubits")

    bad = [p for p in pattern if p not in VALID_PAULIS]
    if bad:
        raise ValueError(f"Invalid Pauli labels in pattern: {bad}")


def sample_pauli(p: float, rng: random.Random) -> str:
    """
    Sample a single-qubit depolarizing error:
        I with probability 1 - p
        X with probability p / 3
        Y with probability p / 3
        Z with probability p / 3

    Role in pipeline:
        Implements the physical error model used before syndrome
        extraction and decoding.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must satisfy 0 <= p <= 1, got {p}")

    r = rng.random()

    if r < (1.0 - p):
        return "I"

    threshold_x = (1.0 - p) + p / 3.0
    threshold_y = (1.0 - p) + 2.0 * p / 3.0

    if r < threshold_x:
        return "X"
    if r < threshold_y:
        return "Y"
    return "Z"


def sample_error_pattern(
    qubits: Sequence[int],
    p: float,
    rng: random.Random,
) -> List[str]:
    """
    Sample an independent Pauli error for each target qubit.

    Role in pipeline:
        Produces the multi-qubit error pattern applied to the encoded
        circuit during channel simulation.
    """
    validate_qubits(qubits)
    return [sample_pauli(p, rng) for _ in qubits]


def apply_single_pauli(qc: QuantumCircuit, qubit: int, pauli: str) -> None:
    """
    Append one sampled Pauli error to a circuit.

    Role in pipeline:
        Converts an abstract sampled error into the circuit operation
        that corrupts the encoded state.
    """
    if pauli == "I":
        return
    if pauli == "X":
        qc.x(qubit)
        return
    if pauli == "Y":
        qc.y(qubit)
        return
    if pauli == "Z":
        qc.z(qubit)
        return
    raise ValueError(f"Invalid Pauli '{pauli}'")


def apply_pattern_to_circuit(
    qc: QuantumCircuit,
    qubits: Sequence[int],
    pattern: Sequence[str],
) -> List[AppliedError]:
    validate_qubits(qubits)
    validate_pattern(qubits, pattern)

    applied: List[AppliedError] = []

    for qubit, pauli in zip(qubits, pattern):
        apply_single_pauli(qc, qubit, pauli)
        if pauli != "I":
            applied.append(AppliedError(qubit=qubit, pauli=pauli))

    return applied


def format_pattern(qubits: Sequence[int], pattern: Sequence[str]) -> str:
    validate_pattern(qubits, pattern)
    return ", ".join(f"q{q}:{p}" for q, p in zip(qubits, pattern))


def make_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed)
