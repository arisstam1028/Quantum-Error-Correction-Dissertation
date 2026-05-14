# Table Decoding And Error Correction

This section turns stabilizer generators into syndrome-measurement circuits and a hard-decision syndrome table.

## Files

### `stabilizer_measurement.py`

General stabilizer parser and measurement-circuit builder.

Functions/classes:

- `StabilizerMeasurementResult`: stores an error, algebraic syndrome, measured syndrome, and match flag.
- `StabilizerParser.from_strings(stabilizers)`: validates Pauli strings and commutation.
- `StabilizerParser.from_symplectic_rows(rows)`: converts `[X | Z]` rows to Pauli strings.
- `StabilizerParser.validate_commutation(stabilizers)`: checks all stabilizers commute.
- `StabilizerParser.pauli_strings_commute(p, q)`: returns whether two strings commute.
- `StabilizerParser.pauli_strings_anticommute(p, q)`: counts local anticommutations.
- `StabilizerParser.single_qubit_anticommutes(a, b)`: checks single-qubit Pauli anticommutation.
- `StabilizerParser.make_single_qubit_error(pauli, qubit, n)`: creates an error string.
- `StabilizerMeasurementBuilder`: builds ancilla-based stabilizer measurement circuits.
- `StabilizerMeasurementBuilder.build_measurement_only_circuit(...)`: builds only syndrome extraction.
- `StabilizerMeasurementBuilder.append_to_encoded_circuit(...)`: appends measurement to an encoder circuit.

### `decoder.py`

Syndrome-table decoder.

Functions/classes:

- `DecodedError`: stores the Pauli and qubit selected as correction.
- `SyndromeTableDecoder`: maps syndrome bitstrings to one correction.
- `SyndromeTableDecoder._build_table()`: computes the identity and all single-qubit X/Y/Z syndromes.
- `SyndromeTableDecoder.decode(syndrome)`: returns the table correction for a syndrome.

### `main.py`

Standalone table-decoding demonstration.

Functions:

- `simulate_measured_syndrome(qc)`: executes a circuit and extracts syndrome bits.
- `print_section(title)`: terminal formatting helper.
- `main()`: builds stabilizer measurement, applies test errors, and checks table decoding.

### `plotter.py`

Circuit and metric plotting helper.

Classes:

- `StabilizerCircuitPlotter`: draws circuits and error-rate curves.

## Theory

For a distance-3 five-qubit code, every single-qubit Pauli error has a distinct syndrome. The table decoder therefore corrects all weight-1 errors by mapping each syndrome to a representative correction.

