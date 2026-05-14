# New4

This is the most complete classical LDPC experiment section. It uses a fixed `(500,1000)` regular `(3,6)` parity-check matrix and explores MS, NMS, OMS, BER, FER, and BER-versus-iteration curves.

## Shared Assumptions

Most scripts in this folder import:

```python
from ldpc_H_matrix import H
```

They expect:

```text
H.shape = (500, 1000)
column weight = 3
row weight = 6
rate approximately 1/2
```

The channel is BPSK over AWGN, with all-zero codeword transmission.

## Files

### `ldpc_H_matrix.py`

Stores the fixed parity-check matrix.

### `new4(a=1).py`

Min-sum with `alpha=1`, equivalent to unnormalised min-sum.

Functions:

- `nzsign(x)`: non-zero sign helper.
- `horizontal_step_min_sum(H, Q, alpha=1)`: check update.
- `vertical_step(H, R, q)`: variable update.
- `min_sum_decode_single(...)`: decodes one frame.
- `simulate_min_sum_ber(...)`: runs BER simulation.

### `new4(Eb-N0 = -2).py`

Variant of `new4` configured for a different Eb/N0 range.

Functions are the same as `new4(a=1).py`, but with `alpha=0.8` and altered experiment settings.

### `new5.py` and `new5(0.3).py`

Main BER/FER comparison scripts for MS/NMS/OMS.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step_nms(Q, alpha=0.8)`: normalized min-sum check update.
- `horizontal_step_oms(Q, beta=0.15)`: offset min-sum check update.
- `vertical_step(R, q)`: variable update using precomputed edge mask.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `decode_single(...)`: decodes one frame using selected mode: `MS`, `NMS`, or `OMS`.
- `simulate_ldpc(...)`: runs Monte Carlo simulation with max frames, minimum frames, and error-limit stopping.
- `plot_curves(...)`: plots BER and FER against Eb/N0, with uncoded BPSK reference.

### `new6.py`

Compact decoder abstraction for MS/NMS/OMS.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step(H, Q, mode="NMS", alpha=0.8, beta=0.15)`: dispatches check-node update based on decoder mode.
- `vertical_step(R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `decode_single(...)`: decodes one frame.
- `simulate_ldpc(...)`: runs BER simulation.

### `new7.py`

Another BER simulator with explicit `H` arguments.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step_min_sum(H, Q, mode="NMS", alpha=0.8, beta=0.15)`: supports min-sum variants.
- `vertical_step(H, R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `min_sum_decode_single(...)`: decodes one frame.
- `simulate_ldpc_ber(...)`: estimates BER.

### `new8.py`

Single-SNR BER-versus-iteration script.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step_nms(Q, alpha=0.8)`: normalized min-sum check update.
- `vertical_step(R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `decode_single_with_iter_ber(...)`: returns BER after each iteration for one frame.
- `simulate_ldpc_and_iter_curve(...)`: averages iteration curves over many frames.

### `new9.py`

BER-versus-iteration curves for one or more SNR points.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step_nms(Q, alpha=0.8)`: check update.
- `vertical_step(R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `decode_single_with_iter_errors(...)`: records bit errors after each iteration.
- `simulate_ldpc_with_iter_curves(...)`: accumulates per-iteration BER curves.
- `plot_ber_curve(...)`: plots final BER.
- `plot_iter_curves(...)`: plots BER against iteration.

### `new10.py`

Multi-Eb/N0 BER-versus-iteration plotter.

Functions:

- `nzsign(x)`: sign helper.
- `horizontal_step_nms(Q, alpha=0.8)`: check-node update.
- `vertical_step(R, q)`: variable update.
- `compute_q_hat(R, q)`: posterior LLR calculation.
- `decode_single_with_iter_errors(...)`: records bit errors at each iteration.
- `simulate_iteration_curve(...)`: averages the iteration curve for one Eb/N0 point.

