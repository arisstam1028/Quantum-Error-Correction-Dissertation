# Core

This section contains reusable binary, Pauli, CSS, and syndrome helpers.

## Files

### `css.py`

CSS matrix helpers.

Functions:

- `css_commutation_check(Hx, Hz)`: returns whether `Hx Hz^T = 0 mod 2`.
- `build_full_symplectic_check(Hx, Hz)`: builds a full binary symplectic check matrix from CSS X/Z checks.

### `helpers.py`

GF(2) linear algebra helpers.

Functions/classes:

- `_gf2_row_echelon_with_pivots(A)`: row-reduces a binary matrix, returning echelon form, rank, and pivot columns.
- `gf2_row_echelon(A)`: public row-echelon wrapper.
- `gf2_rank(A)`: computes rank over GF(2).
- `ensure_binary_matrix(A)`: validates that a matrix contains only 0/1 entries.
- `binary_vector_to_str(v)`: formats a binary vector as a string.
- `GF2RowSpaceChecker`: precomputes a row-space basis and tests membership efficiently.
- `GF2RowSpaceChecker.contains(v)`: returns whether `v` lies in the row space.
- `in_rowspace(v, H)`: one-shot row-space membership helper.

### `pauli.py`

Pauli/binary conversion.

Functions:

- `pauli_string_to_binary(pauli)`: converts a Pauli string to `(ex, ez)`.
- `binary_to_pauli_string(ex, ez)`: converts binary arrays back to a Pauli string.
- `add_pauli_errors(ex1, ez1, ex2, ez2)`: XORs two Pauli errors modulo global phase.

### `syndrome.py`

Syndrome helpers.

Functions:

- `compute_css_syndrome(Hx, Hz, ex, ez)`: returns `(sX, sZ)`.
- `compute_full_syndrome(Hx, Hz, ex, ez)`: concatenates `sX` and `sZ`.
- `batch_css_syndrome(Hx, Hz, ex_batch, ez_batch)`: vectorized syndrome computation.
- `syndrome_matches(H, estimated_error, target_syndrome)`: checks a binary parity equation.

