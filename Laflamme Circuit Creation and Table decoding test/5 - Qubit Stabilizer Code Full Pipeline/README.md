# Five-Qubit Stabilizer Code Full Pipeline

This folder is the integrated five-qubit stabilizer pipeline. It combines encoder construction, stabilizer measurement, syndrome-table decoding, depolarizing noise, and Monte Carlo simulation.

## Files

### `main.py`

Top-level demonstration script.

Functions:

- `simulate_syndrome(qc)`: runs a Qiskit circuit and extracts a measured syndrome bitstring.
- `pattern_to_error(pattern)`: formats a sampled Pauli pattern into a string.
- `count_errors(error)`: counts non-identity Paulis in an error string.
- `print_section(title)`: prints labelled terminal sections.
- `print_symplectic_matrix(title, rows)`: displays binary symplectic rows as `X | Z`.
- `run_single_iteration_mode()`: performs one end-to-end circuit-level demonstration.
- `main()`: chooses the simulation mode and runs the pipeline.

### `simulation_runner.py`

Vectorized Monte Carlo simulator for the five-qubit code.

Classes and functions:

- `SweepResult`: stores probabilities, requested frames, used frames, failures, and logical error rates.
- `SimulationRunner.print_section(title)`: terminal formatting helper.
- `SimulationRunner.print_symplectic_matrix(title, rows)`: prints `[X | Z]` rows.
- `SimulationRunner.pauli_strings_to_xz(stabilizers)`: converts stabilizer strings to binary X/Z matrices.
- `SimulationRunner.build_decoder_lookup_arrays(decoder, n, r)`: turns the syndrome table into an array indexed by syndrome integer.
- `SimulationRunner.sample_error_codes(batch_size, n, p)`: vectorized symmetric depolarizing-channel sampler using integer Pauli codes.
- `SimulationRunner.codes_to_xz(codes)`: converts integer Pauli codes to binary symplectic arrays.
- `SimulationRunner.syndrome_bits_from_errors(ex, ez, sx, sz)`: computes all syndrome bits using symplectic products.
- `SimulationRunner.syndrome_bits_to_indices(bits)`: maps syndrome bit rows to integer table indices.
- `SimulationRunner.residual_identity_mask(error_codes, correction_codes)`: checks whether correction and error multiply to identity up to global phase.
- `SimulationRunner.run_error_rate_sweep_vectorized(...)`: runs a chunked Monte Carlo sweep with early stopping by failures.
- `SimulationRunner.run()`: builds the encoder/stabilizers/table and runs the sweep.

### `stabilizer_config.py`

Stabilizer-basis selector.

Functions:

- `get_hs_derived_stabilizers()`: builds the encoder bundle, reads its `Hs`, and converts rows to Pauli strings.
- `get_active_stabilizers(use_paper_stabilizers=False)`: returns either the Hs-derived basis or the paper's cyclic five-qubit stabilizers.

### `syndrome_table_test.py`

Small circuit-level test of the syndrome table.

Functions/classes:

- `Row`: stores one expected syndrome-table test case.
- `simulate(qc)`: runs a Qiskit circuit and returns measured syndrome.
- `make_error(pauli, q, n=5)`: creates a single-qubit Pauli error.
- `run_syndrome_table_test()`: verifies table entries against circuit measurement.

## Subsections

- `Encoding_Circuit_Builder`: Algorithm 1 encoder construction and verification.
- `Logical_Operator_Calculation`: GF(2) and symplectic tools for logical operators.
- `Table_Decoding_and_Error_Correction`: stabilizer parsing, measurement circuits, and table decoding.
- `Depolarizing_Channel`: circuit-level Pauli error sampling and application.

