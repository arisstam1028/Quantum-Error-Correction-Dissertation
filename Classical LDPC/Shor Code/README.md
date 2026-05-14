# Shor Code

This section contains quantum-error-correction background scripts and diagrams. It is not part of the classical LDPC Monte Carlo pipeline, but it helps bridge from classical parity-check ideas to stabilizer codes.

## Files

### `Shor.py`

Builds the Shor `(9,1)` encoder circuit.

Functions:

- `shor_encode(alpha=1.0, beta=0.0)`: normalizes the input amplitudes, initializes qubit 0 in `alpha|0> + beta|1>`, spreads phase-flip protection across three blocks, then applies bit-flip repetition inside each block.

### `Shor Stabalizers.py`

Draws a stabilizer graph for the Shor code.

Functions:

- `draw_shor_stabilizer_graph(save_path=None, dpi=600)`: creates a visual diagram of stabilizer relationships.

### `Depolarizing Channel.py`

Draws a depolarizing-channel branch diagram.

Functions:

- `depolarizing_branch_diagram(save_path=None, dpi=600)`: shows the four Pauli outcomes: identity, X, Y, and Z.

### `Depolarising channel Vertical.py`

Alternative vertical layout for the depolarizing-channel diagram.

Functions:

- `depolarizing_right_angle_clean(save_path=None, dpi=600)`: renders a right-angle/vertical channel diagram.

### `Depolarising and Pauli.py`

Combined depolarizing and Pauli-error visualisation.

Functions:

- `depolarizing_channel_final(save_path=None, dpi=600)`: renders the final visual version used for explanation.

### `Tanner Graph LDPC.py`

Classical Tanner graph visualisation.

Functions:

- `make_regular_ldpc_H(n, m, dv, dc, seed=0, max_tries=5000)`: builds a regular LDPC parity-check matrix.
- `plot_tanner_graph(H)`: draws the bipartite check-variable graph.

## Theory

The Shor code protects one logical qubit using nine physical qubits. It combines bit-flip repetition with phase-flip repetition. The diagram scripts are explanatory aids for Pauli errors, depolarization, stabilizers, and Tanner graphs.

