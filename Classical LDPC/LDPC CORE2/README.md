# LDPC CORE2

This is a more developed classical LDPC pipeline. It introduces fixed matrix generation, PEG-style construction, layered normalized min-sum, and plotting helpers.

## Files

### `generate_peg_H.py`

Builds and exports a fixed parity-check matrix.

Functions:

- `build_peg_ldpc(n=1000, col_w=3, row_w=6, seed=12345)`: constructs a regular LDPC matrix using a PEG-like edge placement heuristic. It fills variable-node edges while preferring lower-degree checks and avoiding obvious short-cycle patterns.
- `write_H_to_py(H, filename="ldpc_H_matrix.py")`: writes the generated matrix into a Python module as `H = np.array(...)`.

### `ldpc_H_matrix.py`

Stores the generated fixed parity-check matrix `H`. This makes simulation results reproducible.

### `ldpc_core.py`

Layered normalized min-sum simulator.

Functions:

- `hard_decision(q_hat)`: maps LLRs to bits.
- `syndrome(H, c)`: computes `H c^T mod 2`.
- `build_neighbors(H)`: builds `checks_to_vars` and `vars_to_checks` adjacency lists from the matrix.
- `min_sum_decode_layered(...)`: decodes one frame using layered scheduling. Each check update immediately updates the affected variable posterior LLRs, which usually converges faster than flooding.
- `simulate_min_sum_ber(...)`: runs a Monte Carlo Eb/N0 sweep with early stopping by target bit errors.

### `ldpc_plots.py`

Plotting utilities.

Functions:

- `plot_ber_vs_iteration_all_snr(ebn0_dB, ber_vs_iter)`: plots per-iteration BER curves.
- `main()`: runs the simulation and displays/saves plots.

### `new2.py`

Experimental normalized min-sum script with explicit horizontal and vertical steps.

Functions:

- `nzsign(x)`: returns `+1` for zero/non-negative and `-1` for negative values.
- `horizontal_step_min_sum(H, Q, alpha=0.8)`: normalized min-sum check update.
- `vertical_step(H, R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `min_sum_decode_single(...)`: decodes one frame.
- `simulate_min_sum_ber(...)`: runs BER simulation.

### `new_21_01_26.py`

Earlier version of `new2.py` without the same normalization parameter treatment.

Functions:

- `nzsign(x)`: non-zero sign helper.
- `horizontal_step_min_sum(H, Q)`: check-node min-sum update.
- `vertical_step(H, R, q)`: variable-node update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `min_sum_decode_single(...)`: decodes one frame.
- `simulate_min_sum_ber(...)`: runs BER simulation.

