# LDPC1

This is the first compact sum-product decoding example. It uses a small parity-check matrix so the full message-passing process can be inspected by eye.

## Files

### `LDPC1.py`

Runs one SPA decoding demonstration.

Functions:

- `safe_atanh(x)`: clips `x` into the open interval `(-1, 1)` before applying `arctanh`, preventing infinities in the check-node update.
- `hard_decision(llr)`: converts posterior LLRs into bits.
- `compute_syndrome(H, c)`: checks whether a hard-decision word satisfies all parity checks.
- `horizontal_step(H, Q)`: implements the SPA check-node update. For each edge `(i,j)`, it multiplies `tanh(Q[i,j']/2)` over all neighbouring variables except `j`, then maps back with `2 atanh`.
- `vertical_step(H, R, q)`: implements the variable-node update. Each outgoing variable-to-check message is the channel LLR plus all incoming check messages except the target check.
- `update_llr(H, R, q)`: computes the final posterior LLR for each bit.
- `run_ldpc_spa()`: builds the toy example, runs the message updates, and passes the results to the display function.

### `display.py`

Tkinter display helper.

Functions:

- `show_results_popup(...)`: receives matrices, channel samples, LLRs, messages, syndromes, and decoded words, then renders them in a scrollable popup. It does not perform decoding; it only presents the state produced by `LDPC1.py`.

## Theory

This section demonstrates exact belief propagation on a Tanner graph. Variable nodes represent bits, check nodes represent parity constraints, and messages carry LLRs. A zero syndrome means the decoder has found a vector inside the code space.

