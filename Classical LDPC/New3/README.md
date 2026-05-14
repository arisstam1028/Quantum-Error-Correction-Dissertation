# New3

This section is an intermediate stage between the first LDPC core and the final `New4` experiments. It focuses on matrix generation and normalized min-sum decoding.

## Files

### `new_H_generator.py`

Improved PEG-style parity-check generator.

Functions:

- `build_peg_ldpc(n=1000, col_w=3, row_w=6, seed=12345, max_tries_per_edge=5000)`: constructs a sparse regular matrix while attempting to avoid repeated edges and short cycles.
- `write_H_to_py(H, filename="ldpc_H_matrix.py")`: exports `H` as a Python file.

### `ldpc_H_matrix.py`

Stores a generated fixed `H` matrix.

### `new3.py`

Normalized min-sum simulation using the fixed matrix.

Functions:

- `nzsign(x)`: sign helper with `sign(0)=+1`.
- `horizontal_step_min_sum(H, Q, alpha=0.8)`: computes normalized min-sum check messages.
- `vertical_step(H, R, q)`: computes variable-to-check messages.
- `min_sum_decode_single(H, y, sigma2, max_iter=50, alpha=0.8)`: decodes one received frame.
- `simulate_min_sum_ber(H, ...)`: runs BER simulation for the supplied matrix.

## Role In The Dissertation Code

This directory is mainly developmental. It tests the matrix construction and decoding ideas that are made more complete in `New4`.

