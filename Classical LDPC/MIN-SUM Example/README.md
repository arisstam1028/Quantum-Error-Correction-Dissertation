# MIN-SUM Example

This section develops min-sum LDPC decoding from a simple pedagogical script into faster simulation implementations.

## Theory

The min-sum decoder approximates the SPA check-node update:

```text
R_ij = product(sign(Q_ij')) * min(|Q_ij'|)
```

where the product and minimum exclude the target edge. This is cheaper than the `tanh/atanh` SPA update and is standard in LDPC simulations.

## Files

### `Min_Sum.py`

Small readable min-sum demonstration.

Functions:

- `hard_decision(llr)`: converts LLRs to bits.
- `compute_syndrome(H, c)`: computes the parity-check syndrome.
- `horizontal_step_min_sum(H, Q)`: computes check-to-variable min-sum messages.
- `vertical_step(H, R, q)`: computes variable-to-check messages.
- `update_llr(H, R, q)`: computes posterior LLRs.
- `run_ldpc_min_sum(iterations=2, use_random_noise=False)`: runs the toy example for a fixed number of iterations.

### `display.py`

Contains `show_results_popup(...)`, which displays the min-sum matrices and decoded output in a Tkinter window.

### `ldpc_min_sum_ber.py`

First full Monte Carlo BER version.

Functions:

- `build_regular_h(...)`: constructs a random regular LDPC parity-check matrix with fixed row and column weights.
- `hard_decision_from_llr(q_hat)`: converts LLRs to hard decisions.
- `syndrome(H, v)`: computes `H v^T mod 2`.
- `horizontal_step_min_sum(H, Q)`: loop-based check-node update.
- `vertical_step(H, R, q)`: variable-node update.
- `compute_q_hat(H, R, q)`: posterior LLR calculation.
- `min_sum_decode_single(...)`: decodes one noisy frame.
- `simulate_min_sum_ber(...)`: sweeps Eb/N0 and estimates BER.

### `ldpc_min_sum_ber_vectorised.py`

Vectorised version of the BER simulator.

Functions:

- `build_regular_h(...)`: creates the regular `H` matrix.
- `horizontal_step_min_sum(H, Q)`: vectorises sign and minimum calculations across rows.
- `vertical_step(H, R, q)`: updates variable messages.
- `compute_q_hat(H, R, q)`: computes posterior LLRs.
- `min_sum_decode_single(...)`: decodes one frame.
- `simulate_min_sum_ber(...)`: runs BER simulations.
- `_self_test_small()`: sanity-checks the implementation on a small code.

### `ldpc_min_sum_numpy.py`

NumPy/sparse-adjacency implementation that avoids storing dense messages for all non-edges.

Functions:

- `build_regular_h(...)`: returns `H`, `checks_to_vars`, and `vars_to_checks`.
- `min_sum_decode_numpy(...)`: decodes using adjacency lists and prefix/suffix minima.
- `simulate_numpy(...)`: runs an Eb/N0 sweep and prints BER.

### `ldpc_min_sum_numba.py`

Numba-oriented implementation using flattened adjacency arrays.

Functions:

- `build_regular_h_adj(...)`: builds adjacency lists directly.
- `adjlists_to_flat(...)`: converts adjacency lists to flat arrays and pointer offsets.
- `horizontal_min_sum_numba(...)`: JIT-friendly check-node update.
- `vertical_numba(...)`: JIT-friendly variable-node update.
- `compute_qhat_numba(...)`: computes posterior LLRs.
- `syndrome_numba(...)`: checks parity constraints.
- `decode_many_frames_numba(...)`: decodes many frames inside a compiled loop.
- `simulate_numba(...)`: runs the BER experiment.

### `Fastest.py`

Compact fast min-sum experiment.

Functions:

- `build_regular_h(...)`: constructs a regular parity-check matrix.
- `hard_decision(q_hat)`: maps LLRs to bits.
- `syndrome(H, c)`: computes the syndrome.
- `min_sum_decode(...)`: decodes one frame.
- `simulate_min_sum_ber(...)`: runs an Eb/N0 sweep.

