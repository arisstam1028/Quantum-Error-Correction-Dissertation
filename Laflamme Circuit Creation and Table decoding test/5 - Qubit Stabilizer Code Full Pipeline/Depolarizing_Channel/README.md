# Depolarizing Channel

This section provides circuit-level Pauli error sampling and plotting for the five-qubit code.

## Files

### `pauli.py`

Pauli sampling and circuit application helpers.

Functions/classes:

- `AppliedError`: records a qubit index and Pauli label.
- `validate_qubits(qubits)`: checks that qubit indices are valid.
- `validate_pattern(qubits, pattern)`: checks that a sampled pattern matches the target qubits.
- `sample_pauli(p, rng)`: samples `I`, `X`, `Y`, or `Z` from a depolarizing distribution.
- `sample_error_pattern(qubits, p, rng)`: samples one Pauli for each qubit.
- `apply_single_pauli(qc, qubit, pauli)`: appends the corresponding Qiskit gate.
- `apply_pattern_to_circuit(qc, qubits, pattern)`: applies a full sampled pattern.
- `format_pattern(qubits, pattern)`: creates a readable error string.
- `make_rng(seed=None)`: creates a Python random generator.

### `depolarizing_channel.py`

Object wrapper around Pauli error sampling.

Classes:

- `DepolarizingChannel`: stores a probability, target qubits, and RNG seed; samples/appends Pauli errors to circuits.

### `main.py`

Standalone channel experiment.

Functions:

- `import_encoder_builder()`: imports the encoder builder from the sibling section.
- `simulate_observed_error_counts(...)`: repeatedly samples the channel and counts observed errors.
- `main()`: runs the demonstration and plots results.

### `plotter.py`

Plotting helper.

Classes:

- `ChannelPlotter`: visualises observed error counts and channel behaviour.

