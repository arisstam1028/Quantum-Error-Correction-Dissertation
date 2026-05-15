from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PAULIS = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class DecodedError:
    pauli: str
    qubit: Optional[int]

    def correction_text(self) -> str:
        if self.pauli == "I" or self.qubit is None:
            return "No correction required"
        return f"Apply {self.pauli} on qubit {self.qubit}"


class FiveQubitTableDecoder:
    """
    Table decoder for the 5-qubit stabilizer code.

    The syndrome table is built automatically from the stabilizer generators
    by checking commutation / anti-commutation of each single-qubit Pauli error
    with each stabilizer generator.

    Syndrome bit convention:
        bit  0  -> commutes with stabilizer
        bit  1  -> anti-commutes with stabilizer

    Syndrome order:
        (M1, M2, M3, M4)
    """

    def __init__(self) -> None:
        self.stabilizers: List[str] = [
            "XZZXI",  # M1
            "IXZZX",  # M2
            "XIXZZ",  # M3
            "ZXIXZ",  # M4
        ]

        self.n = len(self.stabilizers[0])
        self.table: Dict[str, DecodedError] = self._build_table()

    def _build_table(self) -> Dict[str, DecodedError]:
        table: Dict[str, DecodedError] = {}

        no_error_syndrome = self.compute_syndrome("I" * self.n)
        table[no_error_syndrome] = DecodedError("I", None)

        for qubit in range(self.n):
            for pauli in ("X", "Y", "Z"):
                error_string = self.make_single_qubit_error(pauli, qubit, self.n)
                syndrome = self.compute_syndrome(error_string)

                decoded = DecodedError(pauli=pauli, qubit=qubit)

                if syndrome in table:
                    existing = table[syndrome]
                    raise ValueError(
                        "Duplicate syndrome detected while building the table: "
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
        self._validate_pauli_string(error, expected_len=self.n)

        bits = []
        for stabilizer in self.stabilizers:
            anticommutes = self.pauli_strings_anticommute(error, stabilizer)
            bits.append("1" if anticommutes else "0")

        return "".join(bits)

    def decode(self, syndrome: str) -> DecodedError:
        if syndrome not in self.table:
            raise ValueError(f"Unknown syndrome: {syndrome}")
        return self.table[syndrome]

    def decode_error_string(self, error: str) -> Tuple[str, DecodedError]:
        syndrome = self.compute_syndrome(error)
        decoded = self.decode(syndrome)
        return syndrome, decoded

    def print_generated_table(self) -> None:
        print("Generated 5-qubit syndrome table")
        print("(syndrome -> error)")
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
            if FiveQubitTableDecoder.single_qubit_anticommutes(a, b):
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


if __name__ == "__main__":
    decoder = FiveQubitTableDecoder()

    decoder.print_generated_table()

    print("\nExample decodes:\n")

    examples = ["IIIII", "XIIII", "IZIII", "IIYII", "IIIIX"]
    for error in examples:
        syndrome, decoded = decoder.decode_error_string(error)
        print(f"Error      : {error}")
        print(f"Syndrome   : {syndrome}")
        print(f"Decoded as : {decoded}")
        print(f"Correction : {decoded.correction_text()}")
        print()