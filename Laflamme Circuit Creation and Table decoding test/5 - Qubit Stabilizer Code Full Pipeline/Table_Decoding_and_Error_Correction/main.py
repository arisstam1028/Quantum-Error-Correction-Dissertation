from __future__ import annotations

from qiskit_aer import AerSimulator

from Encoding_Circuit_Builder.main import build_five_qubit_encoder_bundle
from .decoder import SyndromeTableDecoder
from .stabilizer_measurement import StabilizerMeasurementBuilder, StabilizerParser
from .plotter import StabilizerCircuitPlotter


def simulate_measured_syndrome(qc) -> str:
    """
    Run the circuit and return the measured syndrome in (M1,M2,...,Mr) order.
    """
    backend = AerSimulator()
    result = backend.run(qc, shots=1).result()
    counts = result.get_counts()

    if len(counts) != 1:
        raise ValueError(f"Expected a deterministic syndrome, got counts: {counts}")

    counts_key = next(iter(counts.keys()))
    return StabilizerMeasurementBuilder.counts_key_to_syndrome(counts_key)


def print_section(title: str) -> None:
    print('\n' + '' * 80)
    print(title)
    print('' * 80)


def main() -> None:
    bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")
    encoder_circuit = bundle["simplified_circuit"]
    Hs = bundle["Hs"]

    stabilizers = StabilizerParser.from_symplectic_rows(Hs)

    print_section("1. STABILIZERS TAKEN FROM ENCODER BUILDER")
    for i, stab in enumerate(stabilizers, start=1):
        print(f'M{i}  {stab}')

    print_section("2. GENERAL STABILIZER-MEASUREMENT CIRCUIT BUILDER")
    builder = StabilizerMeasurementBuilder(stabilizers)
    decoder = SyndromeTableDecoder(stabilizers)

    measurement_only = builder.build_measurement_only_circuit()
    print("Measurement-only circuit:")
    print(measurement_only.draw(output="text"))

    print_section("3. INSTANTIATE FOR THE 5-QUBIT ENCODED STATE")
    full_encoded_plus_measurement = builder.append_to_encoded_circuit(encoder_circuit)
    print("Encoded circuit followed by stabilizer measurement:")
    print(full_encoded_plus_measurement.draw(output="text"))

    sample_error = "XIIII"
    sample_error_circuit = builder.append_error_and_measure(
        encoder_circuit,
        sample_error,
        name=f"sample_{sample_error}",
        barrier=True,
    )

    print_section("4. COMPARE MEASURED SYNDROME AGAINST ALGEBRAIC SYNDROME")

    all_match = True
    n = builder.n

    test_errors = ["I" * n]
    for qubit in range(n):
        for pauli in ("X", "Y", "Z"):
            test_errors.append(StabilizerParser.make_single_qubit_error(pauli, qubit, n))

    for error in test_errors:
        algebraic_syndrome = builder.compute_algebraic_syndrome(error)
        test_circuit = builder.append_error_and_measure(
            encoder_circuit,
            error,
            name=f"test_{error}",
            barrier=True,
        )
        measured_syndrome = simulate_measured_syndrome(test_circuit)
        matches = (algebraic_syndrome == measured_syndrome)

        if not matches:
            all_match = False

        print(f"Error             : {error}")
        print(f"Algebraic syndrome: {algebraic_syndrome}")
        print(f"Measured syndrome : {measured_syndrome}")
        print(f"Match             : {matches}")
        print("-" * 60)

    print_section("5. CONNECT TO TABLE DECODING")
    if all_match:
        print("All measured syndromes match the algebraic syndromes.\n")
    else:
        print("There was at least one mismatch.\n")

    for error in test_errors:
        syndrome = builder.compute_algebraic_syndrome(error)
        decoded = decoder.decode(syndrome)

        print(f"Actual error   : {error}")
        print(f"Syndrome       : {syndrome}")
        print(f"Decoded as     : {decoded}")
        print(f"Correction     : {decoded.correction_text()}")
        print("-" * 60)

    print_section("6. PLOT CIRCUITS")
    print("Opening circuit plots...")
    StabilizerCircuitPlotter.show_measurement_circuits(
        measurement_only=measurement_only,
        encoded_plus_measurement=full_encoded_plus_measurement,
        sample_error_circuit=sample_error_circuit,
    )


if __name__ == "__main__":
    main()