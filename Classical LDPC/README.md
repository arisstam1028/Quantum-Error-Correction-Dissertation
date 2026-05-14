# Classical LDPC Project

This project contains the classical-code groundwork used before the quantum-code simulations. The code starts with small belief-propagation examples, then moves to min-sum LDPC decoding, Monte Carlo BER/FER experiments, fixed PEG-style parity-check matrices, and a small Shor-code visualisation section.

## Theory

An LDPC code is specified by a sparse binary parity-check matrix `H`. A bit vector `c` is a valid codeword when:

```text
H c^T = 0 mod 2
```

The code is simulated over an AWGN channel using BPSK:

```text
0 -> +1
1 -> -1
LLR_j = 2 y_j / sigma^2
sigma^2 = 1 / (2 R Eb/N0)
```

The early examples use the sum-product algorithm, where check-node updates use `tanh` and `atanh`. Later examples use min-sum approximations, replacing the exact check-node update with signs and minimum magnitudes. The most developed variants use normalized min-sum (NMS) and offset min-sum (OMS).

## Sections

### `main.py`

Small sum-product algorithm demonstration with a hand-written `3 x 6` parity-check matrix. It opens a Tkinter popup showing each decoding step.

Functions:

- `safe_atanh(x)`: clips values before `arctanh` to avoid numerical overflow.
- `hard_decision(llr)`: maps positive LLRs to `0` and negative LLRs to `1`.
- `compute_syndrome(H, c)`: computes `H c^T mod 2`.
- `horizontal_step(H, Q)`: check-node SPA update using products of `tanh(Q/2)`.
- `vertical_step(H, R, q)`: variable-node update using channel LLR plus incoming check messages.
- `update_llr(H, R, q)`: computes final posterior LLRs.
- `run_ldpc_spa_popup()`: builds the toy example, runs one SPA update, and displays the intermediate matrices.

### `LDPC1`

First standalone SPA example. It separates the decoding logic from display code.

### `MIN-SUM Example`

Min-sum decoding experiments. These files test progressively faster implementations: direct loops, vectorised NumPy, sparse adjacency lists, and Numba.

### `LDPC CORE`

Cleaner classical LDPC simulation. It builds a random regular parity-check matrix and runs min-sum BER experiments.

### `LDPC CORE2`

More mature LDPC simulation. It introduces a fixed hard-coded parity-check matrix, PEG-style construction, layered normalized min-sum, plotting, and iteration-curve reporting.

### `New3`

Intermediate PEG/min-sum experiments. This section refines parity-check generation and normalized min-sum decoding.

### `New4`

Main classical LDPC experiment section. It uses the fixed `(500,1000)` parity-check matrix, compares MS/NMS/OMS ideas, records BER/FER, and plots BER versus iteration.

### `Shor Code`

Quantum-code background visualisations. It includes the Shor `(9,1)` encoder and diagrams for depolarizing channels, stabilizers, and Tanner graphs.

## How To Read This Project

The code is chronological. The later directories are usually more complete than the earlier ones. For final dissertation-style classical LDPC results, `LDPC CORE2` and `New4` are the most relevant sections.

