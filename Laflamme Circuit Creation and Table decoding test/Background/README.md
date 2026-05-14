# Background

This directory contains exploratory and reference scripts used before the integrated five-qubit pipeline was assembled.

## Sections

### `Algorithm1`, `Algorithm1_1`, `Algorithm1_Beta`

Successive versions of the stabilizer encoder builder. The later versions contain the same core ideas as the integrated `Encoding_Circuit_Builder`: validation of binary symplectic matrices, GF(2) rank, controlled-Pauli construction, CZ audits, circuit simplification, and encoder verification.

### `Basis Encoder`

Tests a direct basis-state encoder.

Files:

- `basis_encoder.py`: builds encoded basis states/circuits.
- `main.py`: runs the basis-encoder demonstration.

### `Logical Operator Calculation`

Development versions of the logical-operator pipeline.

Files:

- `css_converter.py`: converts CSS-style matrices into stabilizer/symplectic form.
- `css_logicals.py`: computes logical operators for CSS codes.
- `stabilizer_logicals.py`, `stabilizer_logicals_v2.py`, `stabilizer_logicals_v3.py`, `stabilizer_logicals_v4.py`: successive versions of the general stabilizer logical-operator code.
- `main.py`: demonstration entrypoint.

### `Simple QLDPC bicycle CSS code`

Small bicycle-code experiments.

Files:

- `bicycle.py`: builds commuting CSS bicycle matrices, Tanner graphs, degree histograms, anticommutation graphs, and girth estimates.
- `bicycleQ.py`: converts bicycle CSS matrices to Pauli stabilizers and visualises Tanner/commutation graphs.

## Role

These scripts are best treated as notebooks in code form. They preserve the development path and theory experiments, while the integrated pipeline folders contain the cleaner final structure.

