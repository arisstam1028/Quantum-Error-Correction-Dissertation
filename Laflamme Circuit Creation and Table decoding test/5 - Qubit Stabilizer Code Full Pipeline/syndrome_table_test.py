from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from qiskit_aer import AerSimulator

from Encoding_Circuit_Builder.main import build_five_qubit_encoder_bundle
from Table_Decoding_and_Error_Correction.stabilizer_measurement import (
    StabilizerMeasurementBuilder,
)
from stabilizer_config import get_active_stabilizers


# Paper Table 1
EXPECTED_SYNDROMES: Dict[Tuple[str, int], str] = {
    ("X", 0): "0001",
    ("Z", 0): "1010",
    ("Y", 0): "1011",

    ("X", 1): "1000",
    ("Z", 1): "0101",
    ("Y", 1): "1101",

    ("X", 2): "1100",
    ("Z", 2): "0010",
    ("Y", 2): "1110",

    ("X", 3): "0110",
    ("Z", 3): "1001",
    ("Y", 3): "1111",

    ("X", 4): "0011",
    ("Z", 4): "0100",
    ("Y", 4): "0111",
}


@dataclass
class Row:
    error: str
    expected: str
    measured: str
    passed: bool


def simulate(qc):
    sim = AerSimulator()
    result = sim.run(qc, shots=1).result()
    counts = result.get_counts()
    key = next(iter(counts.keys()))
    return key[::-1]


def make_error(pauli, q, n=5):
    s = ["I"] * n
    s[q] = pauli
    return "".join(s)


def run_syndrome_table_test():

    bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")
    encoder = bundle["simplified_circuit"]

    stabilizers = get_active_stabilizers(use_paper_stabilizers=True)
    builder = StabilizerMeasurementBuilder(stabilizers)

    tests = list(EXPECTED_SYNDROMES.keys())

    print("=" * 80)
    print("PAPER SYNDROME TABLE TEST")
    print("=" * 80)

    passed_all = True

    for pauli, q in tests:

        error = make_error(pauli, q)

        qc = builder.append_error_and_measure(
            encoder,
            error,
            barrier=True,
        )

        measured = simulate(qc)
        expected = EXPECTED_SYNDROMES[(pauli, q)]

        ok = (measured == expected)
        passed_all &= ok

        print(
            ("PASS" if ok else "FAIL"),
            "|",
            error,
            "| expected",
            expected,
            "| got",
            measured,
        )

    print("-" * 80)
    print("OVERALL:", "PASS" if passed_all else "FAIL")