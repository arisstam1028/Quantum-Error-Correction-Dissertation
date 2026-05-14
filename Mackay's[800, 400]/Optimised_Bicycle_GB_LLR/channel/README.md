# Channel

This section samples physical Pauli errors for the QLDPC Monte Carlo simulations.

## Files

### `depolarizing.py`

Defines the depolarizing channel.

Classes:

- `DepolarizingChannel`: dataclass storing an optional random seed and NumPy random generator.

Methods:

- `__post_init__()`: creates `self.rng`.
- `sample_error(n, p)`: samples an `n`-qubit Pauli error and returns `(ex, ez)` arrays. It maps `X -> (1,0)`, `Y -> (1,1)`, `Z -> (0,1)`, and `I -> (0,0)`.
- `sample_error_batch(batch_size, n, p)`: samples a batch by repeatedly calling `sample_error`.
- `channel_binary_prior(p)`: returns the binary prior `(p0, p1)` used by CSS BP, where `p1 = 2p/3`.

## Theory

For a symmetric depolarizing channel, an X component appears for X or Y errors, and a Z component appears for Z or Y errors. Therefore each binary component has marginal probability `2p/3`.

