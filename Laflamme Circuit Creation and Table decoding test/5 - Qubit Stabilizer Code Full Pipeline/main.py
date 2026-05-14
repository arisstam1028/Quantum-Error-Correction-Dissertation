from __future__ import annotations

from qiskit_aer import AerSimulator

from simulation_runner import SimulationRunner
from syndrome_table_test import run_syndrome_table_test

from Encoding_Circuit_Builder.main import build_five_qubit_encoder_bundle
from Depolarizing_Channel.depolarizing_channel import DepolarizingChannel
from Table_Decoding_and_Error_Correction.decoder import SyndromeTableDecoder
from Table_Decoding_and_Error_Correction.stabilizer_measurement import (
    StabilizerMeasurementBuilder,
)
from Table_Decoding_and_Error_Correction.plotter import StabilizerCircuitPlotter

from stabilizer_config import get_active_stabilizers


# ============================
# GLOBAL MODES
# ============================

simulate = True
test_syndrome_table = False

# 🔑 THIS is the important switch
use_paper_stabilizers_for_measurement = True


# ============================
# UTILITIES
# ============================

def simulate_syndrome(qc) -> str:
    backend = AerSimulator()
    result = backend.run(qc, shots=1).result()
    counts = result.get_counts()

    if len(counts) != 1:
        raise ValueError(f"Expected exactly one counts outcome, got: {counts}")

    key = next(iter(counts.keys()))
    return key[::-1]


def pattern_to_error(pattern: list[str]) -> str:
    return "".join(pattern)


def count_errors(error: str) -> int:
    return sum(1 for p in error if p != "I")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_symplectic_matrix(title: str, rows: list[list[int]]) -> None:
    if not rows:
        print(f"{title}: <empty>")
        return

    n = len(rows[0]) // 2
    print(title)

    for row in rows:
        x = "".join(str(b) for b in row[:n])
        z = "".join(str(b) for b in row[n:])
        print(f"  {x} | {z}")


# ============================
# SINGLE RUN
# ============================

def run_single_iteration_mode() -> None:
    print_section("SINGLE-ITERATION MODE")

    bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")

    encoder_object = bundle["encoder"]
    encoder_circuit = bundle["simplified_circuit"]
    Hs = bundle["Hs"]

    spec = encoder_object.spec
    n = spec.n
    k = spec.k
    r = spec.r

    print(f"n = {n}")
    print(f"k = {k}")
    print(f"r = {r}")

    print_section("Hs MATRIX")
    print_symplectic_matrix("Hs (X | Z):", Hs)

    print_section("INITIAL STABILIZERS")

    stabilizers = get_active_stabilizers(
        use_paper_stabilizers=use_paper_stabilizers_for_measurement
    )

    for i, stab in enumerate(stabilizers, start=1):
        print(f"M{i} = {stab}")

    print_section("BUILD SYNDROME MEASUREMENT + DECODER")

    builder = StabilizerMeasurementBuilder(stabilizers)
    decoder = SyndromeTableDecoder(stabilizers)

    syndrome_only_circuit = builder.build_measurement_only_circuit()
    encoded_plus_syndrome_circuit = builder.append_to_encoded_circuit(
        encoder_circuit,
        name="encoded_plus_syndrome",
    )

    print_section("ENCODER CIRCUIT")
    print(encoder_circuit.draw(output="text"))

    print_section("SYNDROME-ONLY CIRCUIT")
    print(syndrome_only_circuit.draw(output="text"))

    print_section("ENCODER + SYNDROME CIRCUIT")
    print(encoded_plus_syndrome_circuit.draw(output="text"))

    print_section("PASS THROUGH DEPOLARIZING CHANNEL")

    channel = DepolarizingChannel(p=0.3, seed=32)
    qubits = list(range(encoder_circuit.num_qubits))
    pattern = channel.sample_error_pattern(qubits)

    error = pattern_to_error(pattern)
    weight = count_errors(error)

    print(f"Sampled error pattern: {error}")
    print(f"Error weight: {weight}")

    print_section("APPLY ERROR + MEASURE SYNDROME")

    error_and_syndrome_circuit = builder.append_error_and_measure(
        encoder_circuit,
        error,
        name="encoded_error_syndrome",
        barrier=True,
    )

    syndrome_expected = builder.compute_algebraic_syndrome(error)
    syndrome_measured = simulate_syndrome(error_and_syndrome_circuit)

    print(f"Expected syndrome: {syndrome_expected}")
    print(f"Measured syndrome: {syndrome_measured}")
    print(f"Syndromes match: {syndrome_expected == syndrome_measured}")

    print_section("DECODE")

    decoded = decoder.decode(syndrome_measured)

    print(f"Decoder output: {decoded}")
    print(f"Correction: {decoded.correction_text()}")

    print_section("INTERPRETATION")

    if weight == 0:
        print("No error occurred.")
    elif weight == 1:
        print("Single-qubit error occurred.")
        print("The 5-qubit code guarantees correction.")
    else:
        print("More than one physical error occurred.")
        print("Decoder may be incorrect.")

    print_section("ENCODER + ERROR + SYNDROME CIRCUIT")
    print(error_and_syndrome_circuit.draw(output="text"))

    StabilizerCircuitPlotter.show_measurement_circuits(
        encoder_circuit=encoder_circuit,
        measurement_only=syndrome_only_circuit,
        encoded_plus_measurement=encoded_plus_syndrome_circuit,
        sample_error_circuit=error_and_syndrome_circuit,
    )


# ============================
# MAIN
# ============================

def main() -> None:
    if test_syndrome_table:
        print("\nSyndrome-table test mode enabled.\n")
        run_syndrome_table_test()
        return

    if simulate:
        print("\nMonte Carlo simulation mode enabled.\n")
        SimulationRunner.run()
    else:
        print("\nSingle-iteration mode enabled.\n")
        run_single_iteration_mode()


if __name__ == "__main__":
    main()