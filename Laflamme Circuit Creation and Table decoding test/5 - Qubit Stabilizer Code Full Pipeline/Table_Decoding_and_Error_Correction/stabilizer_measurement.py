# Purpose:
#   Parse stabilizer generators and build syndrome-measurement circuits.
#
# Process:
#   1. Validate Pauli-string or binary symplectic stabilizer input.
#   2. Check that all stabilizer generators commute.
#   3. Build ancilla-based circuits that measure each stabilizer.
#
# Theory link:
#   Stabilizer measurement extracts an error syndrome: which stabilizer
#   constraints are violated. It detects errors without directly
#   measuring or destroying the logical quantum state.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from qiskit import QuantumCircuit


PAULIS = {"I", "X", "Y", "Z"}


@dataclass(frozen=True)
class StabilizerMeasurementResult:
    error: str
    algebraic_syndrome: str
    measured_syndrome: str
    matches: bool


class StabilizerParser:
    """
    General stabilizer parser.

    Supports:
    1. Pauli-string stabilizers, e.g. ["XZZXI", "IXZZX", ...]
    2. Binary symplectic rows from Hq, where each row is [x | z]
    """

    @staticmethod
    def from_strings(stabilizers: Sequence[str]) -> List[str]:
        """
        Validate and normalize Pauli-string stabilizers.

        Role in pipeline:
            Ensures that syndrome extraction is built from a commuting
            stabilizer set.
        """
        if not stabilizers:
            raise ValueError("stabilizers must not be empty")

        n = len(stabilizers[0])
        if n == 0:
            raise ValueError("stabilizers must not contain empty strings")

        cleaned = []
        for s in stabilizers:
            s_clean = s.strip().upper()
            if len(s_clean) != n:
                raise ValueError("All stabilizers must have the same length")
            bad = [ch for ch in s_clean if ch not in PAULIS]
            if bad:
                raise ValueError(f"Invalid Pauli symbols in stabilizer {s}: {bad}")
            cleaned.append(s_clean)

        StabilizerParser.validate_commutation(cleaned)
        return cleaned

    @staticmethod
    def from_symplectic_rows(rows: Sequence[Sequence[int]]) -> List[str]:
        """
        Convert binary symplectic rows [X | Z] to Pauli strings.

        Role in pipeline:
            Connects encoder output Hs to the circuit-based stabilizer
            measurement builder.
        """
        if not rows:
            raise ValueError("rows must not be empty")

        row_len = len(rows[0])
        if row_len == 0 or row_len % 2 != 0:
            raise ValueError("Each symplectic row must have even length 2n")

        n = row_len // 2
        stabilizers = []

        for row in rows:
            if len(row) != row_len:
                raise ValueError("All symplectic rows must have the same length")
            if any(bit not in (0, 1) for bit in row):
                raise ValueError("Symplectic rows must contain only 0/1 values")

            x = row[:n]
            z = row[n:]
            chars = []
            for xi, zi in zip(x, z):
                if xi == 0 and zi == 0:
                    chars.append("I")
                elif xi == 1 and zi == 0:
                    chars.append("X")
                elif xi == 0 and zi == 1:
                    chars.append("Z")
                elif xi == 1 and zi == 1:
                    chars.append("Y")
                else:
                    raise ValueError("Unexpected binary symplectic entry")
            stabilizers.append("".join(chars))

        StabilizerParser.validate_commutation(stabilizers)
        return stabilizers

    @staticmethod
    def validate_commutation(stabilizers: Sequence[str]) -> None:
        """
        Confirm that every pair of stabilizer generators commutes.

        Role in pipeline:
            Commutation is required so all stabilizers can be measured
            consistently as one stabilizer code.
        """
        for i in range(len(stabilizers)):
            for j in range(i + 1, len(stabilizers)):
                if not StabilizerParser.pauli_strings_commute(stabilizers[i], stabilizers[j]):
                    raise ValueError(
                        f"Stabilizers do not commute: {stabilizers[i]} and {stabilizers[j]}"
                    )

    @staticmethod
    def pauli_strings_commute(p: str, q: str) -> bool:
        return not StabilizerParser.pauli_strings_anticommute(p, q)

    @staticmethod
    def pauli_strings_anticommute(p: str, q: str) -> bool:
        if len(p) != len(q):
            raise ValueError("Pauli strings must have the same length")

        anticomm_count = 0
        for a, b in zip(p, q):
            if StabilizerParser.single_qubit_anticommutes(a, b):
                anticomm_count += 1

        return (anticomm_count % 2) == 1

    @staticmethod
    def single_qubit_anticommutes(a: str, b: str) -> bool:
        if a not in PAULIS or b not in PAULIS:
            raise ValueError(f"Invalid Pauli pair: ({a}, {b})")
        if a == "I" or b == "I":
            return False
        if a == b:
            return False
        return True

    @staticmethod
    def make_single_qubit_error(pauli: str, qubit: int, n: int) -> str:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid error Pauli: {pauli}")
        if not (0 <= qubit < n):
            raise ValueError(f"Qubit index {qubit} out of range for n={n}")

        chars = ["I"] * n
        chars[qubit] = pauli
        return "".join(chars)


class StabilizerMeasurementBuilder:
    """
    General ancilla-based stabilizer measurement circuit builder.

    For r stabilizers on n data qubits:
    - uses n data qubits
    - adds r ancilla qubits
    - measures each ancilla into one classical syndrome bit

    Syndrome bit ordering is:
        (M1, M2, ..., Mr)
    """

    def __init__(self, stabilizers: Sequence[str]) -> None:
        self.stabilizers = StabilizerParser.from_strings(stabilizers)
        self.n = len(self.stabilizers[0])
        self.r = len(self.stabilizers)

    def build_measurement_only_circuit(self, *, name: str = "syndrome_measurement") -> QuantumCircuit:
        """
        Build only the ancilla-based stabilizer measurement part.

        Circuit layout:
            data qubits   : 0 .. n-1
            ancilla qubits: n .. n+r-1
            classical bits: 0 .. r-1
        """
        qc = QuantumCircuit(self.n + self.r, self.r, name=name)
        data_qubits = list(range(self.n))
        ancilla_qubits = list(range(self.n, self.n + self.r))

        for stab_index, stabilizer in enumerate(self.stabilizers):
            anc = ancilla_qubits[stab_index]
            # Each ancilla accumulates the parity/eigenvalue of one stabilizer.
            self._append_single_stabilizer_measurement(
                qc,
                stabilizer=stabilizer,
                ancilla=anc,
                data_qubits=data_qubits,
                classical_bit=stab_index,
            )

        return qc

    def append_to_encoded_circuit(
        self,
        encoded_circuit: QuantumCircuit,
        *,
        name: str = "encoded_plus_syndrome",
    ) -> QuantumCircuit:
        """
        Create a new circuit containing:
            encoded data circuit
            + ancilla-based stabilizer measurements

        Assumes encoded_circuit acts on exactly n data qubits.
        """
        if encoded_circuit.num_qubits != self.n:
            raise ValueError(
                f"Encoded circuit has {encoded_circuit.num_qubits} qubits, "
                f"but stabilizers act on {self.n} qubits"
            )

        full_qc = QuantumCircuit(self.n + self.r, self.r, name=name)

        # Put the encoded circuit on the data qubits only
        full_qc.compose(encoded_circuit, qubits=list(range(self.n)), inplace=True)

        # Append stabilizer measurements
        data_qubits = list(range(self.n))
        ancilla_qubits = list(range(self.n, self.n + self.r))

        for stab_index, stabilizer in enumerate(self.stabilizers):
            anc = ancilla_qubits[stab_index]
            self._append_single_stabilizer_measurement(
                full_qc,
                stabilizer=stabilizer,
                ancilla=anc,
                data_qubits=data_qubits,
                classical_bit=stab_index,
            )

        return full_qc

    def append_error_and_measure(
        self,
        encoded_circuit: QuantumCircuit,
        error: str,
        *,
        name: str = "encoded_error_syndrome",
        barrier: bool = True,
    ) -> QuantumCircuit:
        """
        Build:
            encoded circuit
            -> specified Pauli error on data qubits
            -> stabilizer measurement circuit

        error is an n-qubit Pauli string such as:
            "XIIII", "IIYII", "IIIIZ", ...
        """
        self._validate_pauli_string(error, expected_len=self.n)

        full_qc = QuantumCircuit(self.n + self.r, self.r, name=name)

        # 1) compose encoder on data qubits
        full_qc.compose(encoded_circuit, qubits=list(range(self.n)), inplace=True)

        if barrier:
            full_qc.barrier()

        # 2) apply the Pauli error to data qubits
        self.apply_pauli_error(full_qc, error, data_qubits=list(range(self.n)))

        if barrier:
            full_qc.barrier()

        # 3) append syndrome measurement
        data_qubits = list(range(self.n))
        ancilla_qubits = list(range(self.n, self.n + self.r))

        for stab_index, stabilizer in enumerate(self.stabilizers):
            anc = ancilla_qubits[stab_index]
            self._append_single_stabilizer_measurement(
                full_qc,
                stabilizer=stabilizer,
                ancilla=anc,
                data_qubits=data_qubits,
                classical_bit=stab_index,
            )

        return full_qc

    def compute_algebraic_syndrome(self, error: str) -> str:
        """
        Compute the syndrome algebraically:
            0 if the error commutes with the stabilizer
            1 if the error anti-commutes with the stabilizer
        """
        self._validate_pauli_string(error, expected_len=self.n)

        bits = []
        for stabilizer in self.stabilizers:
            anticommutes = StabilizerParser.pauli_strings_anticommute(error, stabilizer)
            bits.append("1" if anticommutes else "0")
        return "".join(bits)

    @staticmethod
    def apply_pauli_error(qc: QuantumCircuit, error: str, data_qubits: Sequence[int]) -> None:
        if len(error) != len(data_qubits):
            raise ValueError("Error length must match number of data qubits")

        for pauli, qubit in zip(error, data_qubits):
            if pauli == "I":
                continue
            if pauli == "X":
                qc.x(qubit)
            elif pauli == "Y":
                qc.y(qubit)
            elif pauli == "Z":
                qc.z(qubit)
            else:
                raise ValueError(f"Invalid Pauli in error string: {pauli}")

    @staticmethod
    def counts_key_to_syndrome(counts_key: str) -> str:
        """
        Convert a Qiskit counts key into syndrome order (c0,c1,...,c_{r-1}).

        Qiskit returns bitstrings with the highest classical bit on the left,
        so we reverse the string to recover:
            syndrome  (M1, M2, ..., Mr)
        """
        return counts_key[::-1]

    def _append_single_stabilizer_measurement(
        self,
        qc: QuantumCircuit,
        *,
        stabilizer: str,
        ancilla: int,
        data_qubits: Sequence[int],
        classical_bit: int,
    ) -> None:
        """
        Measure one stabilizer using one ancilla.

        Procedure:
            ancilla |0> H controlled-Pauli-string H measure
        """
        qc.h(ancilla)

        for pauli, data_q in zip(stabilizer, data_qubits):
            if pauli == "I":
                continue
            if pauli == "X":
                qc.cx(ancilla, data_q)
            elif pauli == "Y":
                qc.cy(ancilla, data_q)
            elif pauli == "Z":
                qc.cz(ancilla, data_q)
            else:
                raise ValueError(f"Invalid Pauli in stabilizer: {pauli}")

        qc.h(ancilla)
        qc.measure(ancilla, classical_bit)

    @staticmethod
    def _validate_pauli_string(pauli_string: str, expected_len: int) -> None:
        if len(pauli_string) != expected_len:
            raise ValueError(
                f"Pauli string must have length {expected_len}, got {len(pauli_string)}"
            )
        bad = [ch for ch in pauli_string if ch not in PAULIS]
        if bad:
            raise ValueError(f"Invalid Pauli characters in string: {bad}")
