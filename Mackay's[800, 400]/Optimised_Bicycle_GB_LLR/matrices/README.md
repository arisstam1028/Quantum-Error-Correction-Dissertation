# Matrices

This section stores fixed matrix modules used by the simulation.

Every simulation matrix module is expected to define:

```python
C   # base circulant matrix
Hx  # X-check parity matrix
Hz  # Z-check parity matrix
```

## Main Files

### `mackay_800_400.py`

Main dissertation matrix. It defines a MacKay-style bicycle QLDPC code:

```text
N       = 800
K       = 400
RANK    = 200
SUPPORT = [10, 53, 83, 117, 128, 165, 245, 259, 262, 289, 313, 356, 378, 395, 399]
C       = 400 x 400 circulant matrix
Hx, Hz  = 200 x 800 matrices
```

The file sets `Hx = H` and `Hz = H`. The resulting CSS commutation condition holds.

### `bicycle_24.py`, `bicycle_24_12.py`

Small bicycle-code matrices and generation/search helpers. `bicycle_24_12.py` includes:

- `circulant_from_support(m, support)`: builds a circulant matrix from support positions.
- `gf2_rank(A)`: rank over GF(2).
- `css_commutes(Hx, Hz)`: checks CSS commutation.
- `density(H)`: matrix density.
- `prune_rows(H, rows_to_drop)`: removes selected rows.
- `build_bicycle_code(support)`: builds a small bicycle code.
- `code_summary(Hx, Hz)`: computes summary parameters.
- `is_suitable_qldpc(Hx, Hz)`: applies suitability checks.
- `format_matrix_py(name, Mtx)`: serializes a matrix as Python code.
- `export_fixed_matrix_file(...)`: writes a fixed matrix module.
- `search_for_code()`: searches support/pruning choices.
- `print_summary(...)`: prints search results.
- `main()`: runs the search/export process.

### `gb_code_7.py`, `gb_code_15.py`, `gb_code_15_2.py`, `gb_code_15_OG.py`, `gb_code_35.py`

Small generalised bicycle code modules used for testing and comparison.

### `GB_bicycle_code_5_family`, `GB_bicycle_code_5_2_family`, `GB_bicycle_code_5_2_p_family`

Families of related matrix modules used by `QLDPCFamilyRunner`. Their filenames end in `_m<number>.py`, which is how the runner discovers them.

