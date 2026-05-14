# LDPC CORE

This section is the first cleaned-up classical LDPC simulation. It builds a regular LDPC parity-check matrix, transmits the all-zero codeword over AWGN, decodes with min-sum, and plots BER results.

## Files

### `ldpc_core.py`

Core simulation logic.

Functions:

- `build_regular_h(n=1000, col_w=3, row_w=6, max_attempts=5000, seed=12345)`: builds a random regular parity-check matrix. It uses the configuration-model idea of pairing column stubs and row stubs, then rejects duplicate row-column pairs.
- `hard_decision(q_hat)`: converts posterior LLRs into bits.
- `syndrome(H, c)`: computes `H c^T mod 2`.
- `min_sum_decode(H, y, sigma2, max_iter=50, c_true=None, ber_traj=None)`: decodes one received frame. It initializes channel LLRs, performs check-node min-sum updates, variable-node updates, posterior LLR calculation, hard decisions, and syndrome-based early stopping.
- `simulate_min_sum_ber(...)`: runs Monte Carlo simulation across Eb/N0 values. It records BER, average iterations, a per-iteration BER trajectory, and metadata about the code.

### `ldpc_H_matrix.py`

Hard-coded parity-check matrix module. It stores an `H` matrix as a NumPy array so later scripts can reuse a fixed code instead of regenerating a random one.

### `ldpc_plots.py`

Plotting entrypoint for results produced by `ldpc_core.py`.

Functions:

- `main()`: runs the simulation and plots BER/iteration information.

## Theory

This section assumes the all-zero codeword, which is valid for a linear code and sufficient for symmetric BPSK/AWGN simulations. The decoder succeeds when the syndrome of the hard decision becomes zero.

