# Five-Qubit Table Decoding: Independent-BSC Approximation

This folder simulates the five-qubit code with table decoding under a binary symmetric approximation to depolarizing noise.

## Files

### `main.py`

Entrypoint.

Functions:

- `main()`: builds `FiveQubitCode`, creates `SimulationRunner`, runs the configured probability sweep, prints a summary, and plots results.

### `simulation_runner.py`

Main simulation logic.

Classes:

- `BinarySymplectic`: conversion and arithmetic helpers for Pauli strings and `(ex, ez)` arrays.
- `FiveQubitCode`: stores the four stabilizer generators and infers `n_qubits`.
- `SimulationConfig`: stores probabilities, frame counts, RNG seed, and optional failure threshold.
- `SimulationResults`: stores probabilities, frames, logical failure rates, Z-basis QBER, X-basis QBER, and average QBER.
- `SimulationRunner`: samples errors, computes syndromes, table-decodes, forms residuals, and accumulates metrics.
- `SimulationReport`: prints and plots results.

Important methods:

- `BinarySymplectic.pauli_char_to_bits(pauli)`: maps `I/X/Z/Y` to binary pairs.
- `BinarySymplectic.bits_to_pauli_char(x, z)`: converts binary pairs back to Pauli labels.
- `BinarySymplectic.pauli_string_to_bsf(pauli_string)`: converts a string to `(ex, ez)`.
- `BinarySymplectic.bsf_to_pauli_string(ex, ez)`: converts `(ex, ez)` to a Pauli string.
- `BinarySymplectic.add_errors(...)`: XORs two Pauli errors modulo global phase.
- `BinarySymplectic.weight(ex, ez)`: counts non-identity positions.
- `SimulationRunner.run(config)`: performs the Monte Carlo sweep.

### `Depolarizing_Channel/depolarizing.py`

Noise model.

Classes:

- `DepolarizingChannel`: samples binary error components.

Methods:

- `sample_error(n, p)`: if `use_independent_bsc_approx=True`, samples `ex` and `ez` independently with probability `2p/3`; otherwise samples exact symmetric Pauli depolarizing errors.
- `sample_error_batch(batch_size, n, p)`: repeats `sample_error`.
- `channel_binary_prior(p)`: returns `(1 - 2p/3, 2p/3)`.

### `Table_Decoding_and_Error_Correction/stabilizer_measurement.py`

Binary stabilizer measurement.

Classes:

- `BinarySymplectic`: converts Pauli strings to X/Z arrays.
- `StabilizerMeasurement`: validates stabilizers, builds `hx` and `hz`, and computes syndromes.

Methods:

- `compute_syndrome(ex, ez)`: computes `hx @ ez + hz @ ex mod 2`.
- `validate_binary_error(...)`: checks binary vector shape and values.
- `validate_pauli_string(...)`: checks Pauli-string length and symbols.

### `Table_Decoding_and_Error_Correction/decoder.py`

Syndrome table.

Classes:

- `SyndromeTableDecoder`: builds and applies a fixed single-qubit correction table.

Methods:

- `_build_lookup()`: inserts identity and all single-qubit X/Y/Z syndromes.
- `decode(syndrome)`: returns correction arrays.
- `make_single_qubit_error(pauli, qubit, n_qubits)`: creates one binary Pauli error.

### `Table_Decoding_and_Error_Correction/plotter.py`

Plotting helper.

Classes:

- `StabilizerCircuitPlotter`: plots QBER/FER curves.

### `test_depolarizing.py`

Channel sanity check.

Functions:

- `binary_pair_to_pauli(ex, ez)`: maps binary pairs to Pauli labels.
- `test_depolarizing_channel(p, n_qubits=10000, seed=42)`: samples many errors and prints empirical frequencies.

