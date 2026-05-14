# Logical Operator Calculation

This section computes logical operators from stabilizer data using GF(2) linear algebra and binary symplectic products.

## Files

### `stabilizer_logicals.py`

Main logical-operator pipeline.

Classes:

- `GF2`: binary arithmetic helpers, including XOR addition, dot products, and symplectic product.
- `GF2Rank`: row-reduction, rank, and independent-row extraction over GF(2).
- `PauliBinary`: converts between Pauli strings and binary symplectic rows.
- `ElimResult`: stores elimination output.
- `XHalfEliminator`: performs elimination on the X half of a stabilizer matrix.
- `StandardForm`: stores the standard-form matrices and dimensions.
- `StandardFormBuilder`: converts a stabilizer matrix into the form needed to identify logical operators.
- `LogicalOps`: stores logical X and logical Z rows.
- `LogicalOperatorBuilder`: constructs anticommuting logical operator pairs.
- `StabilizerChecks`: verifies commutation and independence conditions.
- `Pretty`: formats binary symplectic rows as Pauli strings.
- `StabilizerPipeline`: end-to-end wrapper for loading stabilizers, standardising them, building logicals, and printing results.

### `main.py`

Demonstration entrypoint.

Functions:

- `demo_with_stabilizers()`: runs the pipeline from Pauli stabilizer strings.
- `demo_with_Hq()`: runs the pipeline from binary symplectic rows.
- `main()`: selects and runs the demo.

## Theory

Logical operators commute with all stabilizers but are not themselves products of stabilizers. For a stabilizer code with `n` physical qubits and `r` independent stabilizers, `k = n - r` logical qubits are encoded. Logical X/Z pairs must anticommute with each other and commute with the stabilizer group.

