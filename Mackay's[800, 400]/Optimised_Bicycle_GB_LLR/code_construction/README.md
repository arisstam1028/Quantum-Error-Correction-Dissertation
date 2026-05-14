# Code Construction

This section contains utilities for building bicycle CSS codes from circulant matrices and analysing their parameters.

## Files

### `circulant.py`

Circulant matrix helpers.

Functions:

- `first_row_from_support(m, support)`: creates a binary first row of length `m` with ones at the support positions.
- `random_sparse_first_row(m, weight, rng=None)`: samples a sparse binary first row of given weight.
- `circulant_from_first_row(first_row)`: builds a circulant matrix by rolling the first row.
- `column_weights(H)`: returns column weights.
- `row_weights(H)`: returns row weights.
- `density(H)`: returns the fraction of ones in the matrix.

### `bicycle_code.py`

Bicycle CSS construction.

Classes/functions:

- `BicycleCode`: stores `C`, `Hx`, `Hz`, and the first row used to generate `C`.
- `build_bicycle_code(...)`: builds a circulant matrix `C`, forms `H0 = [C | C^T]`, sets `Hx = H0`, `Hz = H0`, optionally prunes rows, and checks CSS commutation.

### `code_analysis.py`

Code-statistics helpers.

Classes/functions:

- `CodeStats`: stores block length, check counts, GF(2) ranks, estimated `k`, densities, row weights, and column weights.
- `analyze_css_code(Hx, Hz)`: computes those statistics.
- `print_code_stats(Hx, Hz)`: prints a compact summary.
- `print_matrix(name, M)`: prints one binary matrix.
- `print_bicycle_matrices(C, Hx, Hz)`: prints the bicycle construction matrices.

## Theory

A bicycle code uses a sparse circulant matrix and its transpose to create a structured LDPC parity-check matrix. The CSS commutation condition is:

```text
Hx Hz^T = 0 mod 2
```

