# Purpose:
#   Build a syndrome lookup table for the five-qubit stabilizer code.
#
# Process:
#   1. Compute the syndrome for the identity error.
#   2. Compute syndromes for every single-qubit X, Y, and Z error.
#   3. Map each observed syndrome to one fixed Pauli correction.
#
# Theory link:
#   For a distance-3 code, all weight-one Pauli errors have distinct
#   syndromes. Table decoding corrects by applying a representative
#   Pauli with the matching syndrome.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


PAULIS = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class DecodedError:
    pauli: str
    qubit: Optional[int]

    def correction_text(self) -> str:
        if self.pauli == "I" or self.qubit is None:
            return "No correction required"
        return f"Apply {self.pauli} on qubit {self.qubit}"

    def to_error_string(self, n: int) -> str:
        if self.pauli == "I" or self.qubit is None:
            return "I" * n

        chars = ["I"] * n
        chars[self.qubit] = self.pauli
        return "".join(chars)


class SyndromeTableDecoder:
    """
    General table decoder for a stabilizer code.

    It builds a syndrome table automatically from the provided stabilizer
    generators by enumerating:
        - no error
        - all single-qubit X, Y, Z errors

    This means it is a single-qubit Pauli decoder, which is exactly what
    you want for the 5-qubit code.
    """

    def __init__(self, stabilizers: Sequence[str]) -> None:
        if not stabilizers:
            raise ValueError("stabilizers must not be empty")

        self.stabilizers: List[str] = [s.strip().upper() for s in stabilizers]
        self.n = len(self.stabilizers[0])

        for s in self.stabilizers:
            if len(s) != self.n:
                raise ValueError("All stabilizers must have the same length")
            bad = [ch for ch in s if ch not in PAULIS]
            if bad:
                raise ValueError(f"Invalid stabilizer symbols in {s}: {bad}")

        self.table: Dict[str, DecodedError] = self._build_table()

    def _build_table(self) -> Dict[str, DecodedError]:
        """
        Construct the hard-decision syndrome table.

        Role in pipeline:
            Enumerates all correctable single-qubit Pauli errors and
            stores the correction selected for each syndrome.
        """
        table: Dict[str, DecodedError] = {}

        no_error = "I" * self.n
        no_error_syndrome = self.compute_syndrome(no_error)
        table[no_error_syndrome] = DecodedError("I", None)

        for qubit in range(self.n):
            for pauli in ("X", "Y", "Z"):
                error = self.make_single_qubit_error(pauli, qubit, self.n)
                syndrome = self.compute_syndrome(error)
                decoded = DecodedError(pauli=pauli, qubit=qubit)

                if syndrome in table:
                    existing = table[syndrome]
                    raise ValueError(
                        "Duplicate syndrome detected while building decoder table: "
                        f"{syndrome} maps to both {existing} and {decoded}"
                    )

                table[syndrome] = decoded

        return table

    @staticmethod
    def make_single_qubit_error(pauli: str, qubit: int, n: int) -> str:
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid single-qubit error Pauli: {pauli}")
        if not (0 <= qubit < n):
            raise ValueError(f"Qubit index {qubit} out of range for n={n}")

        chars = ["I"] * n
        chars[qubit] = pauli
        return "".join(chars)

    def compute_syndrome(self, error: str) -> str:
        """
        Compute the syndrome bitstring for a Pauli error.

        Role in pipeline:
            Identifies which stabilizer generators anticommute with the
            error and are therefore violated by it.
        """
        self._validate_pauli_string(error, expected_len=self.n)

        bits = []
        for stabilizer in self.stabilizers:
            anticommutes = self.pauli_strings_anticommute(error, stabilizer)
            bits.append("1" if anticommutes else "0")

        return "".join(bits)

    def decode(self, syndrome: str) -> DecodedError:
        """
        Return the table correction associated with a syndrome.

        Role in pipeline:
            Converts measured stabilizer violations into a Pauli
            correction choice.
        """
        if syndrome not in self.table:
            raise ValueError(
                f"Unknown syndrome: {syndrome}. "
                "This usually means the code is not a single-qubit-perfect lookup case."
            )
        return self.table[syndrome]

    def decode_error_string(self, error: str) -> Tuple[str, DecodedError]:
        syndrome = self.compute_syndrome(error)
        decoded = self.decode(syndrome)
        return syndrome, decoded

    def print_generated_table(self) -> None:
        print("Generated syndrome table")
        print("(syndrome -> decoded single-qubit error)")
        print()

        rows: List[Tuple[str, str]] = [("I" * self.n, self.compute_syndrome("I" * self.n))]
        for qubit in range(self.n):
            for pauli in ("X", "Z", "Y"):
                error = self.make_single_qubit_error(pauli, qubit, self.n)
                syndrome = self.compute_syndrome(error)
                rows.append((error, syndrome))

        for error, syndrome in rows:
            decoded = self.decode(syndrome)
            if decoded.pauli == "I":
                label = "No error"
            else:
                label = f"{decoded.pauli} on qubit {decoded.qubit}"
            print(f"{error} -> {syndrome} -> {label}")

    @staticmethod
    def pauli_strings_anticommute(p: str, q: str) -> bool:
        if len(p) != len(q):
            raise ValueError("Pauli strings must have the same length")

        anticomm_count = 0
        for a, b in zip(p, q):
            if SyndromeTableDecoder.single_qubit_anticommutes(a, b):
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
    def _validate_pauli_string(pauli_string: str, expected_len: int) -> None:
        if len(pauli_string) != expected_len:
            raise ValueError(
                f"Pauli string must have length {expected_len}, got {len(pauli_string)}"
            )

        bad = [ch for ch in pauli_string if ch not in PAULIS]
        if bad:
            raise ValueError(f"Invalid Pauli characters in string: {bad}")
