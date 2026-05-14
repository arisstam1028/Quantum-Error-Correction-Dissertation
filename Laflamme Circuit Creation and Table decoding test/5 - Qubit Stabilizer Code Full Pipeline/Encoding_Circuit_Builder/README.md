# Encoding Circuit Builder

This section builds the five-qubit encoder circuit from a binary symplectic stabilizer matrix using the tutorial-paper Algorithm 1 structure.

## Files

### `Algorithm1.py`

Main encoder implementation.

Functions/classes:

- `_validate_binary_matrix(Hs)`: checks that `Hs` is non-empty, binary, rectangular, and has `2n` columns.
- `_validate_binary_vec(v, length, name)`: validates a binary vector of expected length.
- `_gf2_rank(mat)`: computes rank over GF(2) by Gaussian elimination.
- `_apply_cy_native(qc, ctrl, tgt)`: appends a native Qiskit `CYGate`.
- `_apply_cz_native(qc, ctrl, tgt)`: appends a native Qiskit `CZGate`.
- `EncoderSpec`: stores inferred `n`, `k`, `r`, and stabilizer count.
- `StabilizerEncoder`: builds the encoder from `Hs` and logical-X rows.
- `StabilizerEncoder._apply_pair(...)`: maps a binary `(x,z)` pair to controlled X, Z, or Y logic.
- `StabilizerEncoder.build()`: constructs the full encoder circuit.
- `HsPrinter._format_vec(v)`: formats a symplectic vector as `X | Z`.
- `HsPrinter.print_all(...)`: prints encoder metadata and logical vectors.
- `CircuitPlotter.show(qc, ...)`: draws the Qiskit circuit.

### `main.py`

Five-qubit data and convenience entrypoint.

Functions:

- `get_five_qubit_data()`: returns the five-qubit `Hs` matrix and logical-X vector used by the encoder.
- `build_five_qubit_encoder(...)`: constructs the raw encoder.
- `build_five_qubit_simplified_encoder(...)`: constructs and simplifies the encoder.
- `build_five_qubit_encoder_bundle(...)`: returns a dictionary containing the encoder object, `Hs`, logical-X data, and circuits.
- `main()`: prints/builds/draws the encoder.

### `simplify_encoder_v3.py`

Attempts to remove redundant gates while preserving the stabilizer span.

Functions/classes:

- `SimplifySpec`: configuration for simplification.
- `_stabilizer_span_matches(Hs, qc)`: verifies whether the circuit still generates the required stabilizer span.
- `_remove_gate_indices(qc, remove_idxs)`: creates a circuit with selected gates removed.
- `EncoderSimplifier`: applies simplification passes and checks correctness.
- `SimplifiedCircuitPlotter`: draws the simplified circuit.

### `verify_encoder_v2.py`

Verifies that the encoder maps ancilla Z operators to the expected stabilizer span.

Functions:

- `gf2_rank(mat)`: GF(2) rank.
- `pauli_to_xz(p, n)`: converts Qiskit Pauli data to binary X/Z arrays.
- `make_Z_on_qubit(n, q)`: creates a Pauli Z on one qubit.
- `stabilizers_from_ancilla_Z_images(...)`: propagates ancilla Z operators through the encoder.
- `build_image_rows(...)`: builds binary rows from those images.
- `verify_stabilizer_span_algorithm1(Hs, qc)`: checks rank/span agreement.

### `cz_audit.py`

Audits CZ gates expected from the binary input data.

Functions:

- `expected_cz_edges(Hs, logical_X)`: computes which CZ interactions should appear.
- `actual_cz_edges(qc)`: extracts CZ edges from the circuit.
- `audit_cz(Hs, logical_X, qc)`: compares expected and actual CZ edges.

## Theory

Algorithm 1 uses stabilizer standard form to prepare logical qubits and stabilizer generators. Controlled Pauli operations are chosen from the binary `(x,z)` entries. The verification code checks the construction in the stabilizer/symplectic domain rather than by simulating every statevector.

