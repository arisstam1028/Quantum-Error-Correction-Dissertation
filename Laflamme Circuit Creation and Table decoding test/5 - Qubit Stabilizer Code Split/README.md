# Five-Qubit Stabilizer Code Split

This folder is a split-layout snapshot of the five-qubit stabilizer pipeline. The same ideas appear in the later integrated folder, but here each part was kept as a separate mini-project during development.

## Sections

### `Encoding Circuit Builder`

Builds the stabilizer encoder from Algorithm 1. Its files mirror the integrated `Encoding_Circuit_Builder` section:

- `Algorithm1.py`: validates `Hs`, computes GF(2) rank, and builds controlled-Pauli encoder circuits.
- `main.py`: supplies five-qubit data and builds the encoder.
- `simplify_encoder_v3.py`: removes redundant gates while checking stabilizer-span preservation.
- `verify_encoder_v2.py`: verifies encoder stabilizer images.
- `cz_audit.py`: checks expected versus actual CZ interactions.

### `Depolarizing Channel`

Samples and applies circuit-level Pauli errors.

- `pauli.py`: validates/samples/applies Pauli patterns.
- `depolarizing_channel.py`: channel wrapper.
- `plotter.py`: plots sampled channel data.
- `main.py`: demonstration entrypoint.

### `Logical Operator Calculation`

Computes logical operators from stabilizer data.

- `stabilizer_logicals.py`: GF(2), symplectic, standard-form, and logical-operator tools.
- `main.py`: demonstration entrypoint.

### `Table Decoding and Error Correstion`

Builds stabilizer measurement circuits and syndrome lookup decoding.

- `stabilizer_measurement.py`: parses stabilizers and builds measurement circuits.
- `decoder.py`: maps syndromes to single-qubit Pauli corrections.
- `plotter.py`: circuit plotting.
- `main.py`: demonstration entrypoint.

## Note

The spelling `Error Correstion` is present in the folder name. The integrated pipeline has the corrected, more complete version.

